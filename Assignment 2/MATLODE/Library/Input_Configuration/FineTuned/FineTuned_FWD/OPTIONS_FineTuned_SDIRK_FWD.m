%% OPTIONS_FineTuned_SDIRK_FWD
%
% <html>
%   <div>
%       <img style="float: right" src="../../../../../MATLODE_LOGO.png" height="150px"></img>
%   </div>
% </html>
%
% <html>
% Up: <a href="../../../../html/Library.html">Library</a>
% </html>
%
%% Singly Diagonally Implicit Runge-Kutta (SDIRK) Forward (FWD) Fine Tuning
% The following is the default fine tuning for the SDIRK FWD integrator.
% All option settings fine tuned to 0 will use the default
% configuration settings. 
%
% *IMPORTANT:* If an option setting is required by integrator, the
% corresponding setting must be fined to 0 or an appropriate value as
% desired. 
%
function [ OPTIONS ] = OPTIONS_FineTuned_SDIRK_FWD
    OPTIONS = MATLODE_OPTIONS(...
        'AbsTol',          0, ...
        'ChunkSize',       0, ...
        'displayStats',    0, ...
        'displaySteps',    0, ...
        'FacMax',          0, ...
        'FacMin',          0, ...
        'FacRej',          0, ...
        'FacSafe',         0, ...
        'GMRES_TOL',       0, ...
        'GMRES_MaxIt',     [], ...
        'GMRES_Restart',   [],...
        'GMRES_P',         [],...
        'Hmax',            0, ...
        'Hmin',            0, ...
        'Hstart',          0, ...
        'ITOL',            0, ...
        'LU',              1, ...
        'MatrixFree',      0, ...
        'Max_no_steps',    0, ...
        'Method',          0, ...
        'NewtonMaxIt',     0, ...
        'NewtonTol',       0, ...
        'Qmax',            0, ...
        'Qmin',            0, ...
        'RelTol',          0, ...
        'StartNewton',     0, ...
        'storeCheckpoint', 0, ...
        'ThetaMin',        0 );

end

%%
%  Authored by Tony D'Augustine, Adrian Sandu, and Hong Zhang.
%  Computational Science Laboratory, Virginia Tech.
%  ©2015 Virginia Tech Intellectual Properties, Inc.
%
%%
% <html>
%   <div>
%       <img style="float: right" src="../../../../../CSL_LogoWithName_1.png" height="50px"></img>
%   </div>
% </html>