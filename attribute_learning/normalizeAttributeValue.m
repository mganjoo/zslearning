function [ outputs ] = normalizeAttributeValue( values )
    distinctVals = unique(values);
    assert(length(distinctVals) == 2, 'must have only two values');
    outputs = values;
    outputs(values == distinctVals(1)) = 1;
    outputs(values == distinctVals(2)) = 2;
end

