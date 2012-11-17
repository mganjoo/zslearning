function varargout = stack2param(X, decodeInfo)

assert(length(decodeInfo)==nargout,'this should output as many variables as you gave to get X with param2stack!')

index=0;
for i=1:length(decodeInfo)
    if iscell(decodeInfo{i})
        cellOut = cell(length(decodeInfo{i}),1);
        for c = 1:length(decodeInfo{i})
            matSize = decodeInfo{i}{c};
            cellOut{c} = reshape(X(index+1:index+prod(matSize)),matSize);
            index = index+prod(matSize);
        end
        varargout{i}=cellOut;
    else
        matSize = decodeInfo{i};
        varargout{i} = reshape(X(index+1:index+prod(matSize)),matSize);
        index = index+prod(matSize);
    end
end
