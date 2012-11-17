function retstr = getParamString(params)

fieldNames = fieldnames(params);
retstr = [];
for i = 1:length(fieldNames)
    if strcmp(class(params.(fieldNames{i})), 'function_handle')
        converter = @func2str;
    else
        converter = @mat2str;
    end
    retstr = [retstr fieldNames{i} converter(params.(fieldNames{i})) ','];
end
retstr = retstr(1:end-1);

end