/-- `log` is analytic away from nonpositive reals -/
theorem analyticAt_clog (m : z ∈ slitPlane) : AnalyticAt ℂ log z := by
  /-
    z : Complex
    m : Membership.mem Complex.slitPlane z
    ⊢ AnalyticAt Complex Complex.log z
  -/
  rw [analyticAt_iff_eventually_differentiableAt]
  /-
    z : Complex
    m : Membership.mem Complex.slitPlane z
    ⊢ Filter.Eventually (fun z => DifferentiableAt Complex Complex.log z) (nhds z)
  -/
  filter_upwards [isOpen_slitPlane.eventually_mem m]
  /-
    case h
    z : Complex
    m : Membership.mem Complex.slitPlane z
    ⊢ ∀ (a : Complex), Membership.mem Complex.slitPlane a → DifferentiableAt Compl …
  -/
  intro z m
  /-
    case h
    z✝ : Complex
    m✝ : Membership.mem Complex.slitPlane z✝
    z : Complex
    m : Membership.mem Complex.slitPlane z
    ⊢ DifferentiableAt Complex Complex.log z
  -/
  exact differentiableAt_id.clog m
  /-
    🎉 no goals
  -/


/-- `log` is analytic away from nonpositive reals -/
theorem AnalyticAt.clog (fa : AnalyticAt ℂ f x) (m : f x ∈ slitPlane) :
    AnalyticAt ℂ (fun z ↦ log (f z)) x :=
  (analyticAt_clog m).comp fa


theorem AnalyticWithinAt.clog (fa : AnalyticWithinAt ℂ f s x) (m : f x ∈ slitPlane) :
    AnalyticWithinAt ℂ (fun z ↦ log (f z)) s x :=
  (analyticAt_clog m).comp_analyticWithinAt fa


/-- `log` is analytic away from nonpositive reals -/
theorem AnalyticOnNhd.clog (fs : AnalyticOnNhd ℂ f s) (m : ∀ z ∈ s, f z ∈ slitPlane) :
    AnalyticOnNhd ℂ (fun z ↦ log (f z)) s :=
  fun z n ↦ (analyticAt_clog (m z n)).comp (fs z n)


theorem AnalyticOn.clog (fs : AnalyticOn ℂ f s) (m : ∀ z ∈ s, f z ∈ slitPlane) :
    AnalyticOn ℂ (fun z ↦ log (f z)) s :=
  fun z n ↦ (analyticAt_clog (m z n)).analyticWithinAt.comp (fs z n) m


/-- `f z ^ g z` is analytic if `f z` is not a nonpositive real -/
theorem AnalyticWithinAt.cpow (fa : AnalyticWithinAt ℂ f s x) (ga : AnalyticWithinAt ℂ g s x)
    (m : f x ∈ slitPlane) : AnalyticWithinAt ℂ (fun z ↦ f z ^ g z) s x := by
  have e : (fun z ↦ f z ^ g z) =ᶠ[𝓝[insert x s] x] fun z ↦ exp (log (f z) * g z) := by
    filter_upwards [(fa.continuousWithinAt_insert.eventually_ne (slitPlane_ne_zero m))]
    intro z fz
    simp only [fz, cpow_def, if_false]
  /-
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f g : E → Complex
    x : E
    s : Set E
    fa : AnalyticWithinAt Complex f s x
    ga : AnalyticWithinAt Complex g s x
    m : Membership.mem Complex.slitPlane (f x)
    e : (nhdsWithin x (Insert.insert x s)).EventuallyEq (fun z => HPow.hPow (f z)  …
    ⊢ AnalyticWithinAt Complex (fun z => HPow.hPow (f z) (g z)) s x
  -/
  apply AnalyticWithinAt.congr_of_eventuallyEq_insert _ e
  /-
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f g : E → Complex
    x : E
    s : Set E
    fa : AnalyticWithinAt Complex f s x
    ga : AnalyticWithinAt Complex g s x
    m : Membership.mem Complex.slitPlane (f x)
    e : (nhdsWithin x (Insert.insert x s)).EventuallyEq (fun z => HPow.hPow (f z)  …
    ⊢ AnalyticWithinAt Complex (fun z => Complex.exp (HMul.hMul (Complex.log (f z) …
  -/
  exact ((fa.clog m).mul ga).cexp
  /-
    🎉 no goals
  -/


/-- `f z ^ g z` is analytic if `f z` is not a nonpositive real -/
theorem AnalyticAt.cpow (fa : AnalyticAt ℂ f x) (ga : AnalyticAt ℂ g x)
    (m : f x ∈ slitPlane) : AnalyticAt ℂ (fun z ↦ f z ^ g z) x := by
  /-
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f g : E → Complex
    x : E
    fa : AnalyticAt Complex f x
    ga : AnalyticAt Complex g x
    m : Membership.mem Complex.slitPlane (f x)
    ⊢ AnalyticAt Complex (fun z => HPow.hPow (f z) (g z)) x
  -/
  rw [← analyticWithinAt_univ] at fa ga ⊢
  /-
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f g : E → Complex
    x : E
    fa : AnalyticWithinAt Complex f Set.univ x
    ga : AnalyticWithinAt Complex g Set.univ x
    m : Membership.mem Complex.slitPlane (f x)
    ⊢ AnalyticWithinAt Complex (fun z => HPow.hPow (f z) (g z)) Set.univ x
  -/
  exact fa.cpow ga m
  /-
    🎉 no goals
  -/


/-- `f z ^ g z` is analytic if `f z` avoids nonpositive reals -/
theorem AnalyticOn.cpow (fs : AnalyticOn ℂ f s) (gs : AnalyticOn ℂ g s)
    (m : ∀ z ∈ s, f z ∈ slitPlane) : AnalyticOn ℂ (fun z ↦ f z ^ g z) s :=
  fun z n ↦ (fs z n).cpow (gs z n) (m z n)


/-- `f z ^ g z` is analytic if `f z` avoids nonpositive reals -/
theorem AnalyticOnNhd.cpow (fs : AnalyticOnNhd ℂ f s) (gs : AnalyticOnNhd ℂ g s)
    (m : ∀ z ∈ s, f z ∈ slitPlane) : AnalyticOnNhd ℂ (fun z ↦ f z ^ g z) s :=
  fun z n ↦ (fs z n).cpow (gs z n) (m z n)

