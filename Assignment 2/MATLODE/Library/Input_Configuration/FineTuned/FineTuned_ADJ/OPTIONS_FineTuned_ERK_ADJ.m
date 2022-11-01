%% OPTIONS_FineTuned_ERK_ADJ
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
%% Explicit Runge-Kutta (ERK) Adjoint (ADJ) Fine Tuning
% The following is the default fine tuning for the ERK ADJ integrator.
% All option settings fine tuned to 0 will use the default
% configuration settings. 
%
% *IMPORTANT:* If an option setting is required by integrator, the
% corresponding setting must be fined to 0 or an appropriate value as
% desired.  
%
function OPTIONS = OPTIONS_FineTuned_ERK_ADJ
    OPTIONS = MATLODE_OPTIONS(...
        'AbsTol',          0, ...
        'ChunkSize',       0, ...
        'displayStats',    0, ...
        'displaySteps',    0, ...
        'FacMax',          0, ...
        'FacMin',          0, ...
        'FacRej',          0, ...
        'FacSafe',         0, ...
        'Hmax',            0, ...
        'Hmin',            0, ...
        'Hstart',          0, ...
        'ITOL',            0, ...
        'Max_no_steps',    0, ...
        'Method',          0, ...
        'Qmax',            0, ...
        'Qmin',            0, ...
        'RelTol',          0, ...
        'storeCheckpoint', 0 );

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