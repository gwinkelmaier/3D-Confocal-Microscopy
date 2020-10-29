function Out = myWatershed( I )
    bw = logical(I);
    D = bwdist(~bw);
    D = -D;
    
    m = imextendedmin(D,0.5);
    D = imimposemin(D,m);
    
    D(~bw) = Inf;
    Out = watershed(D);
    Out(~bw) = 0;
    Out(find(Out>0)) = 1;
end
