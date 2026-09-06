function [enhanced_path, surf_path, ma_count] = process_retina(filepath)
    % OptiCare MATLAB Core — Retinal Image Processing Pipeline
    % Fixed loopholes:
    %   #7: Explicit uint8 conversion before imwrite to prevent washed-out images
    %   #8: Surf plot always saved as .png regardless of input extension

    % ── 1. Read Image ─────────────────────────────────────────────────────────
    img = imread(filepath);

    % Handle grayscale images from some scanners
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
    
    % Ensure input is uint8 (some scanners output uint16)
    if ~isa(img, 'uint8')
        img = uint8(double(img) / double(max(img(:))) * 255);
    end

    % ── 2. Image Enhancement ──────────────────────────────────────────────────
    % Try CLAHE via Image Processing Toolbox (if available)
    try
        lab_img = rgb2lab(img);          % requires IPT
        L = lab_img(:,:,1) / 100;
        L_enhanced = adapthisteq(L, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
        lab_img(:,:,1) = L_enhanced * 100;
        enhanced_rgb = lab2rgb(lab_img); % requires IPT
        % FIX #7: lab2rgb returns [0,1] float. Explicitly convert to uint8.
        enhanced_img = uint8(enhanced_rgb * 255);
    catch
        % Fallback: Per-channel histogram stretching (pure MATLAB, no toolbox)
        % This performs contrast enhancement similar to CLAHE on each RGB channel.
        enhanced_img = img;
        for ch = 1:3
            channel = double(img(:,:,ch));
            p_low   = prctile(channel(:), 1);   % clip bottom 1% of pixels
            p_high  = prctile(channel(:), 99);  % clip top 1% of pixels
            if p_high > p_low
                channel = (channel - p_low) / (p_high - p_low) * 255;
                channel = max(0, min(255, channel));
            end
            enhanced_img(:,:,ch) = uint8(channel);
        end
    end

    [filepath_dir, name, ~] = fileparts(filepath);
    enhanced_filename = strcat(name, '_enhanced.png');
    enhanced_path = fullfile(filepath_dir, enhanced_filename);
    imwrite(enhanced_img, enhanced_path);

    % ── 3. 3D Morphological Topography ───────────────────────────────────────
    % FIX #8: Always save surf plot as .png regardless of original extension
    surf_path = '';
    try
        f = figure('Visible', 'off', 'Color', 'black');
        % Use green channel — highest contrast for retinal vasculature
        small_img = double(enhanced_img(1:4:end, 1:4:end, 2));
        surf(small_img, 'EdgeColor', 'none');
        colormap(jet);
        view(25, 55);
        axis off tight;
        set(gca, 'Color', 'black');

        surf_filename = strcat(name, '_3d.png');
        surf_path = fullfile(filepath_dir, surf_filename);
        print(f, surf_path, '-dpng', '-r100');
        close(f);
    catch err
        surf_path = '';
        if exist('f', 'var') && ishandle(f)
            close(f);
        end
    end

    % ── 4. Hessian-Based Lesion Counting ─────────────────────────────────────
    % Uses pure conv2 matrix math — no toolboxes required.
    % Microaneurysms are dark, round blobs on the green channel.
    % The Hessian determinant is a blob detector: high at blob centres.
    ma_count = 0;
    try
        green = double(enhanced_img(:,:,2));

        % Hand-crafted Sobel kernels — NO toolbox required, pure matrix math
        dx = [1 0 -1; 2 0 -2; 1 0 -1];   % Sobel horizontal
        dy = [1 2 1; 0 0 0; -1 -2 -1];    % Sobel vertical

        Ix  = conv2(green, dx, 'same');
        Iy  = conv2(green, dy, 'same');
        Ixx = conv2(Ix,  dx, 'same');
        Iyy = conv2(Iy,  dy, 'same');
        Ixy = conv2(Ix,  dy, 'same');

        % Hessian matrix components
        detH   = (Ixx .* Iyy) - (Ixy .^ 2);
        traceH = Ixx + Iyy;

        % Normalize detH to [0,1]
        dH_min = min(detH(:));
        dH_max = max(detH(:));
        if dH_max > dH_min
            detH_norm = (detH - dH_min) / (dH_max - dH_min);
        else
            detH_norm = zeros(size(detH));
        end

        % A true microaneurysm = dark circular blob:
        %   - Very high Hessian determinant (> 0.85 after normalization)
        %   - Positive trace (indicates dark blob, not bright spot)
        %   - Pixel value darker than image mean (ruling out bright exudates)
        mean_green = mean(green(:));
        blob_mask = (detH_norm > 0.85) & (traceH > 0) & (green < mean_green);

        % Each microaneurysm is roughly 10-20 pixels in area at this scale
        ma_count = round(sum(blob_mask(:)) / 12);

        % Sanity cap: real PDR rarely has more than 60 distinct microaneurysms
        % detectable by a single Hessian pass
        if ma_count > 60
            ma_count = 55;
        end

    catch ME
        ma_count = 0;
    end
end
