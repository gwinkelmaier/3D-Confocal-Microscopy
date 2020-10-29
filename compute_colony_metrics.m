function compute_colony_metrics( file, save_file, thresh )
    %% Load file
    %% Expects variables I, M, P for Image, Mask, and Probabilities
    load(file);
    pred = squeeze(P(:,:,:,2));
    bw = myWatershed(pred>thresh);
    bw = logical(bw);

    %% Nuclear stats
    CC = bwconncomp(bw, 6);
    nuclei = regionprops(CC, 'Area', 'Centroid');

    %% Measure Empty Space
    [~,~,z] = size(bw);
    X = []; Y = []; Z = [];
    for k=1:z
        [x,y] = find(bw(:,:,k));
        X = [X; x];
        Y = [Y; y];
        Z = [Z; k*ones([numel(x), 1])];
    end
    [C, vol] = convhulln([X, Y, Z]);

    empty = numel(bw(bw==1))/vol;

    colony.density = (1 - empty)*100;

    %% Center Of Mass
    centers = [nuclei.Centroid];
    Cx = centers(1:3:end);
    Cy = centers(2:3:end);
    Cz = centers(3:3:end);

    xm = mean(Cx);
    ym = mean(Cy);
    zm = mean(Cz);

    dx = Cx - xm;
    dy = Cy - ym;
    dz = Cz - zm;

    dist = sqrt(dx.^2 + dy.^2 + dz.^2);

    dist = mat2gray(dist);
    colony.pdf = histcounts(dist, 10, 'Normalization', 'probability');

    %% Organization
    perfect_sphere = [zeros([1,9]) 1];
    colony.organization = dot(perfect_sphere, colony.pdf);

    %% Save
    save(save_file, 'nuclei','colony','bw');
end
