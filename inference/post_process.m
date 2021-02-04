function compute_colony_metrics( file, save_file, thresh )
  %%% Post process segmentation results with myWatershed
  %%% and compute initial metrics from binary volume

    %% Load file
    load(file);
    pred = squeeze(P(:,:,:,2));
    bw = myWatershed(pred>thresh);
    bw = logical(bw);

    %% Nuclear stats
    CC = bwconncomp(bw, 6);
    nuclei = regionprops(CC,'Area','Centroid','PixelIdxList','BoundingBox');
    for i=1:numel(nuclei)
        [nuclei(i).x,nuclei(i).y,nuclei(i).z] = ind2sub(size(bw), nuclei(i).PixelIdxList);
    end
    %% Save
    save(save_file, 'nuclei','bw');

end
