% mvpa_analyze.m
%
%    Usage: Runs cross-validation and classification analyses.
%    Author: Akshay Jagadeesh
%    Date: 05/26/2017
%    
%    Inputs:
%       - vw --> view structure that you get from MrVista
%               e.g. vw = MrVista();
%       - roiName --> string containing roiName as it is saved in data/kgs101014_TaskEffects/3Danatomy/ROIs (without the .mat extension)
%
function cvl= mvpa_analyze(vw, roiName, onlyPart2)

if ieNotDefined('roiName')
  % Defaults to VTC if no roiName is passed in
  roiName = 'VTC';
end
if ieNotDefined('onlyPart2')
  onlyPart2 = 0;
end

disp(sprintf('Computing cross-validated MVPA for ROI: %s', roiName));

%Run on selective attention task (scans 4-6)
scanNums = [4 5 6];
scanGroup = 'MotionComp_RefScan1';

% Compute mvpa using this ROI (just loads MVPA if it's already been run)
mv = mv_init(vw, roiName, scanNums, scanGroup);

tSeries = mv.tSeries; % numTimePoints x numVoxels

%Get labels / conditions -- there are 22 unique ones (20 excluding cue and fixation)
labels = mv.trials.label; %1 x numTimePoints --> cell of strings
conds = mv.trials.cond; %1 x numTimePoints --> array of numeric value

%% Generate 2 sets of labels based on which stim is attended / unattended
attLabel = nan(length(conds), 1);
unattLabel = nan(length(conds), 1);

% Remap attended
% 1:FaceA, 2:BodyA, 3:CarA, 4:HouseA, 5:WordA
for i = 1:20
  attLabel(conds==i) = ceil(i/4);
end

% Remap unattended
% 1:FaceU, 2:BodyU, 3:CarU, 4:HouseU, 5:WordU
for i = 1:20
  switch i
    case {5, 9, 13, 17} %
      unattLabel(conds==i) = 1;
    case {1, 10, 14, 18}
      unattLabel(conds==i) = 2;
    case {2, 6, 15, 19}
      unattLabel(conds==i) = 3;
    case{3, 7, 11, 20}
      unattLabel(conds==i) = 4;
    case{4, 8, 12, 16}
      unattLabel(conds==i) = 5;
  end
end

% Throw out the last 3 timepoints because (for some reason) 
%    tSeries is 3 timepoints shorter than the conds/labels.
attLabel = attLabel(1:end-3);
unattLabel = unattLabel(1:end-3);

if onlyPart2
  accuracy = crossCategoryAnalysis(mv, tSeries, conds, attLabel, unattLabel);
  cvl = accuracy;
  return
end

%% Part 1: Compute 1-label classification.
categories = {'Face', 'Body', 'Car', 'House', 'Word'};
crossvalLoss = nan(2,5);
stdError = nan(2,5);
for ci = 1:5
  % Labels: 1 if attending this category, 0 otherwise
  al = attLabel==ci;
  svm = fitcsvm(mv.tSeries, al, 'Crossval', 'on');
  loss = kfoldLoss(svm, 'mode', 'individual');
  crossvalLoss(1,ci) = mean(loss);
  stdError(1,ci) = 1.96*std(loss)/sqrt(length(loss));
  disp(sprintf('%sA Misclassification: %02.02f%%', categories{ci}, 100*mean(loss)))

  % Labels: 1 if unattending this category, 0 otherwise
  ul = unattLabel==ci;
  svm = fitcsvm(mv.tSeries, ul, 'Crossval', 'on');
  uLoss = kfoldLoss(svm, 'mode', 'individual');
  crossvalLoss(2,ci) = mean(uLoss);
  stdError(2,ci) = 1.96*std(uLoss)/sqrt(length(uLoss));
  disp(sprintf('%sU Misclassification: %02.02f%%', categories{ci}, 100*mean(uLoss)));
end

figure; bar(crossvalLoss');
hold on; errorbar((1:5) - .125, crossvalLoss(1,:), stdError(1,:), '.k', 'LineWidth', 1);
errorbar((1:5)+.125, crossvalLoss(2,:), stdError(1,:), '.k', 'LineWidth', 1);
title(sprintf('%s: Within-Condition Crossvalidated Misclassification Rate', roiName));
set(gca, 'XTickLabel', categories);
ylabel('Error');
ylim([0 0.4]);
legend('Attended', 'Unattended');
%drawPublishAxis;

%% Part 2: Compute cross category analysis
accuracy = crossCategoryAnalysis(mv, tSeries, conds, attLabel, unattLabel);

cvl.part1 = crossvalLoss;
cvl.part2 = accuracy;

% crossCategoryAnalysis
%       
%     For each pair of stimuli(e.g. Face,House), trains a 2-way classifier on all attended
%     conditions for those stimuli (e.g. FaceA-CarU, FaceA-BodyU, HouseA-CarU, etc).
%     Then, tests on the shared stimuli (FaceA-HouseU and HouseA-FaceU) to see if it classifies those
%     into the correct attended label.
%
%     Inputs:
%        - mv --> mvpa struct returned from calling 
%
function acc = crossCategoryAnalysis(mv, tSeries, conds, attLabel, unattLabel)

% start with FB/BF
exc = nchoosek(1:5,2);

categories = {'Face', 'Body', 'Car', 'House', 'Word'};

acc = [];
combos = {};
for ei = 1:size(exc,1)
  excluded = exc(ei,:);

  disp(sprintf('Running for excluded %d and %d', excluded(1), excluded(2)));
  combos{end+1} = sprintf('%s-%s', categories{excluded(1)}, categories{excluded(2)});
  traindata = []; trainlabel = []; index = nan(size(tSeries,1),1); count = 0;
  testdata = []; testlabel = []; index2 = nan(size(tSeries,1),1); count2 = 0;
  for i = 1:size(tSeries,1)
    if (any(attLabel(i) == excluded) && ~any(unattLabel(i)==excluded))
      count = count +1;
      traindata(count,:) = tSeries(i,:);
      trainlabel(count) = attLabel(i);
      index(count) = i;
    elseif any(attLabel(i) == excluded) && any(unattLabel(i)==excluded)
      count2 = count2+1;
      testdata(count2,:) = tSeries(i,:);
      testlabel(count2) = attLabel(i);
      index2(count2) = i;
    end
  end
  svm = fitcsvm(traindata, trainlabel');
  label = predict(svm, testdata);
  accuracy = sum(label' == testlabel)/length(testlabel);
  acc(end+1) = accuracy;
  disp(sprintf('Held-out prediction accuracy: %02.02f%%', accuracy*100));

end

figure; bar(acc); hold on;
plot([0 11], [0.5 0.5], '-r');
set(gca, 'XTickLabel', combos);
title(sprintf('%s: Two-Category Classification Accuracy', mv.roi.name));
ylabel('Classification Accuracy');
ylim([0 .8]);
%drawPublishAxis;

