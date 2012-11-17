function tilefigs(handles,resize,nRows,nCols,leftRightSpacing,topBottomSpacing,...
    border,monitor,monitorLocation,monitorSize)
% TILEFIGS Tile figures (spread them out). Tile figs has a number of
% arguments, which may either be specified, or entered as [] to use the
% default
%   TILEFIGS(handles,resize,nRows,nCols,leftRightSpacing,...
%       topBottomSpacing,border,monitor,monitorLocation,monitorSize)
%   TILEFIGS , by itself, resizes and tiles all open figures on to the primary monitor
%   TILEFIGS(handles) Resizes and tiles the figures specified in handles,
%   with the first handle in the upper left corner
%   TILEFIGS(...,resize) If resize is set to false, the figures will be
%   moved but not resized. Effort has been made to have minimal overlap,
%   but some may still occur.
%   TILEFIGS(...,nRows) Specifies a number of rows for the tile grid
%   TILEFIGS(...,nCols) Specifies a number of column for the tile grid.
%   Note that the figures are still included row by row, so if one has 8
%   open figures and tilefigs([],[],[],5) is called, the figures will be
%   put into a 4 x 2 grid with space left for a 5th column.
%   TILEFIGS(...,leftRightSpacing) leaves a horizontal space between the
%   figures. Spacing is specified in pixels. This may have no effect if
%   resize is set to false.
%   TILEFIGS(...,topBottomSpacing) Same as above, except for vertical
%   spacing
%   TILEFIGS(...,border) Leaves a border around the tiling of figures.
%   border is a 1 x 4 matrix, which is [leftSpace bottomSpace rightSpace
%   topSpace]. If the primary monitor is used, the default is [0 35 0 0] 
%   which is approximately the size of the taskbar in Windows 7. If a
%   secondary monitor is used, the default is [0 0 0 0]
%   TILEFIGS(...,monitor , monitorLocation,monitorSize) Specifies a monitor to use, 
%   which is the row of the call get(0,'MonitorPositions'). MATLAB does not
%   officially support dual monitors. The author has found that the
%   MonitorPositions call correctly gets the size of the monitor (in
%   pixels), but not the location of the monitor. monitorLocation and
%   monitorSize overrides the location and size of the monitor.

% Written by Brendan Tracey October 11th, 2012.

% Inspirational credit to:
%Charles Plum                    Nichols Research Corp.
%<cplum@nichols.com>             70 Westview Street
%Tel: (781) 862-9400             Kilnbrook IV
%Fax: (781) 862-9485             Lexington, MA 02173

% Nomenclature -- first number is width, second is length
% x is left to right, y is up and down

if ~exist('topBottomSpacing','var') || isempty(topBottomSpacing)
    topBottomSpacing = 0;
end
if ~exist('leftRightSpacing','var') || isempty(leftRightSpacing)
    leftRightSpacing = 0;
end

%% Select the monitor and get the location and size of the monitor

monitorPositions = get(0,'MonitorPositions');
matlabMonitorLocation = monitorPositions(:,1:2);
matlabMonitorSize = monitorPositions(:,3:4) - monitorPositions(:,1:2) + 1;

if ~exist('monitor','var') || isempty(monitor)
    % No monitor specified, use the primary
    monitor = 1;
end

if ~exist('monitorSize','var') || isempty(monitorSize)
    % No monitor size specified, use what Matlab says
    monitorSize = matlabMonitorSize(monitor,:);
end

if ~exist('monitorLocation','var') || isempty(monitorLocation)
    % No monitor location specified, use what Matlab says
    monitorLocation = matlabMonitorLocation(monitor,1:2);
end

% Set the default border if none is specified [left, bottom, right, top]
if monitor > 1
    if ~exist('border','var') || isempty(border)
        border = [0 0 0 0];
    end
else
    if ~exist('border','var') || isempty(border)
        border = [0 30 0 0];
    end
end
% Modify the monitor size and location to account for the border
monitorLocation = monitorLocation + border(1:2);
monitorSize(1) = monitorSize(1) - border(1) - border(3);
monitorSize(2) = monitorSize(2) - border(2) - border(4);

%% Select figures to use
if ~exist('handles','var') || isempty(handles)
    % No figure handles specified, select all figures
    handles = get (0,'Children'); % locate all open figure handles
    handles = handles(end:-1:1); % Re-order so that first created is in upper left
end
nFigures = length(handles);

%% Determine the grid for figures
if (~exist('nRows','var') || isempty(nRows)) && (~exist('nCols','var') || isempty(nCols))
    % No grid specified, choose the grid which roughly matches the monitor
    % aspect ratio
    monitorAspectRatio = monitorSize(1) / monitorSize(2);
    if monitorAspectRatio < 1
       nCols = round(sqrt(nFigures));
       nRows = ceil(nFigures/nCols);
    else
       nRows = round(sqrt(nFigures));
       nCols = ceil(nFigures/nRows);
    end
elseif (exist('nRows','var') && ~isempty(nRows)) && (~exist('nCols','var') || isempty(nCols))
    nCols = ceil(nFigures/nRows);
elseif (~exist('nRows','var') || isempty(nRows)) && (exist('nCols','var') && ~isempty(nCols))
    nRows = ceil(nFigures/nCols);
elseif (exist('nRows','var') && ~isempty(nRows)) && (exist('nCols','var') && ~isempty(nCols))
    if nRows*nCols < nFigures
        error('Grid size not big enough')
    end
else
    error('Should not be here')
end

%% Calculate the grid sizing
if ~exist('resize','var') || isempty(resize)
    % Resize not set, default is to resize
    resize = 1;
end
if resize
    width = (monitorSize(1) - leftRightSpacing*(nCols - 1))/nCols;
    height = (monitorSize(2) - topBottomSpacing*(nRows - 1))/nRows;
else
    % Make the spacing equal with the constraint that the figures do not go
    % off the edge of the screen.
    figureWidths = zeros(nFigures,1);
    figureHeights = zeros(nFigures,1);
    for ii = 1:nFigures
        figSize = get(handles(ii),'OuterPosition');
        figureWidths(ii) = figSize(3);
        figureHeights(ii) = figSize(4);
    end
    widthEndInds = nCols:nCols:nFigures;
    heightEndInds = 1:nCols;
    maxWidth = max(figureWidths(widthEndInds));
    maxHeight = max(figureHeights(heightEndInds));
    width = (monitorSize(1) - maxWidth)/max(nCols-1,1);
    if nRows ==1
        height = 0;
    else
        height = (monitorSize(2) - maxHeight)/(nRows-1);
    end
end

%% Move and resize the figures
if resize
    pnum = 0;
    for row = 1:nRows
        for col = 1:nCols
            pnum = pnum+1;
            if (pnum>nFigures)
                break
            end
            xLocation = monitorLocation(1) + (col - 1)*width + (col - 1)*leftRightSpacing;
            yLocation = monitorLocation(2) + monitorSize(2) - row*height ...
                - (row - 1)*topBottomSpacing;
            figure(handles(pnum))
            set(handles(pnum),'OuterPosition',[xLocation yLocation width height]);
        end
    end
else
    % Not resizing, set initial locations
    xLocations = inf*ones(nRows,nCols);
    yLocations = inf*ones(nRows,nCols);
    widthMat = zeros(nRows,nCols);
    heightMat = zeros(nRows,nCols);
    pnum = 0;
    for row = 1:nRows
        for col = 1:nCols
            pnum = pnum+1;
            if (pnum>nFigures)
                break
            end
            widthMat(row,col) = figureWidths(pnum);
            heightMat(row,col) = figureHeights(pnum);
        end
    end
    pnum = 0;
    for row = 1:nRows
        for col = 1:nCols
            pnum = pnum+1;
            if (pnum>nFigures)
                break
            end
            xLocation = (col - 1)*width;
            yLocation = monitorSize(2) - (row - 1)*height - heightMat(row,col);
            xLocations(row,col) = xLocation;
            yLocations(row,col) = yLocation;
        end
    end
    
    % Modify the positions to make it look nicer.
    % First, if the figures are too big to not overlap, spread the rows and
    % columns out to use up all of the space at the border to minimize
    % overlap. The subtraction of the width if there are no figures is to 
    % avoid the issue of both the column and the row being moved into the
    % empty spot
    if (sum(sum(widthMat,2) > monitorSize(1)) > 0)
        for row = 1:nRows
            lastNonemptyCol = find(xLocations(row,:) < inf,1,'Last');
            whitespace = monitorSize(1) - xLocations(row,lastNonemptyCol) - widthMat(row,lastNonemptyCol) - width*(nCols-lastNonemptyCol);
            if lastNonemptyCol>1
                for col = 1:lastNonemptyCol
                    xLocations(row,col) = xLocations(row,col) + (col-1)/(lastNonemptyCol-1) * whitespace;
                end
            end
        end
    end
    if sum(sum(heightMat,1) > monitorSize(2)) > 0
        for col = 1:nCols
            lastNonemptyRow = find(xLocations(:,col) < inf,1,'Last');
            whitespace = yLocations(lastNonemptyRow,col) - 1 - height*(nRows - lastNonemptyRow);
            if lastNonemptyRow>1
                for row = 1:lastNonemptyRow
                    yLocations(row,col) = yLocations(row,col) - ((row-1)/(nRows-1)) * whitespace;
                end
            end
        end
    end
    % If the figures can be condensed, slide the rows and columns in.
    for col = 1:nCols - 1
        lastNonemptyRow = find(xLocations(:,col) < inf,1,'Last');
        endOfPlot = xLocations(1:lastNonemptyRow,col) + widthMat(1:lastNonemptyRow,col) + leftRightSpacing;
        beginningOfNextPlot = xLocations(1:lastNonemptyRow,col + 1);
        gap = beginningOfNextPlot - endOfPlot;
        minGap = max(min(gap),0);
        xLocations(1:lastNonemptyRow,col+1) = xLocations(1:lastNonemptyRow,col+1) - minGap;
    end
    for row = 2:nRows
        lastNonemptyCol = find(xLocations(row,:) < inf,1,'Last');
        bottomOfUpperRow = yLocations(row-1,1:lastNonemptyCol) - topBottomSpacing;
        topOfThisRow = yLocations(row,1:lastNonemptyCol) + heightMat(row,1:lastNonemptyCol);
        gap =  bottomOfUpperRow - topOfThisRow;
        minGap = max(min(gap),0);
        yLocations(row,1:lastNonemptyCol) = yLocations(row,1:lastNonemptyCol) + minGap;
    end
    % Finally, actually move and replot the figures.
    pnum = 0;
    for row = 1:nRows
        for col = 1:nCols
            pnum = pnum+1;
            if (pnum>nFigures)
                break
            end
            figure(handles(pnum))
            xloc = xLocations(row,col);
            yloc = yLocations(row,col);
            set(handles(pnum),'OuterPosition',[monitorLocation(1)+xloc monitorLocation(2) + yloc figureWidths(pnum) figureHeights(pnum)]);
        end
    end
end