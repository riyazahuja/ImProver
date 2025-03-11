/-- A map `f` is said to be conformal if it has a conformal differential `f'`. -/
def ConformalAt (f : X → Y) (x : X) :=
  ∃ f' : X →L[ℝ] Y, HasFDerivAt f f' x ∧ IsConformalMap f'


theorem conformalAt_id (x : X) : ConformalAt _root_.id x :=
  ⟨id ℝ X, hasFDerivAt_id _, isConformalMap_id⟩


theorem conformalAt_const_smul {c : ℝ} (h : c ≠ 0) (x : X) : ConformalAt (fun x' : X => c • x') x :=
  ⟨c • ContinuousLinearMap.id ℝ X, (hasFDerivAt_id x).const_smul c, isConformalMap_const_smul h⟩


@[nontriviality]
theorem Subsingleton.conformalAt [Subsingleton X] (f : X → Y) (x : X) : ConformalAt f x :=
  ⟨0, hasFDerivAt_of_subsingleton _ _, isConformalMap_of_subsingleton _⟩


/-- A function is a conformal map if and only if its differential is a conformal linear map -/
theorem conformalAt_iff_isConformalMap_fderiv {f : X → Y} {x : X} :
    ConformalAt f x ↔ IsConformalMap (fderiv ℝ f x) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : NormedAddCommGroup X
    inst✝² : NormedAddCommGroup Y
    inst✝¹ : NormedSpace Real X
    inst✝ : NormedSpace Real Y
    f : X → Y
    x : X
    ⊢ Iff (ConformalAt f x) (IsConformalMap (fderiv Real f x))
  -/
  constructor
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝³ : NormedAddCommGroup X
      inst✝² : NormedAddCommGroup Y
      inst✝¹ : NormedSpace Real X
      inst✝ : NormedSpace Real Y
      f : X → Y
      x : X
      ⊢ ConformalAt f x → IsConformalMap (fderiv Real f x)
    -/
  · rintro ⟨f', hf, hf'⟩
    /-
      case mp.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝³ : NormedAddCommGroup X
      inst✝² : NormedAddCommGroup Y
      inst✝¹ : NormedSpace Real X
      inst✝ : NormedSpace Real Y
      f : X → Y
      x : X
      f' : ContinuousLinearMap (RingHom.id Real) X Y
      hf : HasFDerivAt f f' x
      hf' : IsConformalMap f'
      ⊢ IsConformalMap (fderiv Real f x)
    -/
    rwa [hf.fderiv]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝³ : NormedAddCommGroup X
      inst✝² : NormedAddCommGroup Y
      inst✝¹ : NormedSpace Real X
      inst✝ : NormedSpace Real Y
      f : X → Y
      x : X
      ⊢ IsConformalMap (fderiv Real f x) → ConformalAt f x
    -/
  · intro H
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝³ : NormedAddCommGroup X
      inst✝² : NormedAddCommGroup Y
      inst✝¹ : NormedSpace Real X
      inst✝ : NormedSpace Real Y
      f : X → Y
      x : X
      H : IsConformalMap (fderiv Real f x)
      ⊢ ConformalAt f x
    -/
    by_cases h : DifferentiableAt ℝ f x
      /-
        case pos
        X : Type u_1
        Y : Type u_2
        inst✝³ : NormedAddCommGroup X
        inst✝² : NormedAddCommGroup Y
        inst✝¹ : NormedSpace Real X
        inst✝ : NormedSpace Real Y
        f : X → Y
        x : X
        H : IsConformalMap (fderiv Real f x)
        h : DifferentiableAt Real f x
        ⊢ ConformalAt f x
      -/
    · exact ⟨fderiv ℝ f x, h.hasFDerivAt, H⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : Type u_1
        Y : Type u_2
        inst✝³ : NormedAddCommGroup X
        inst✝² : NormedAddCommGroup Y
        inst✝¹ : NormedSpace Real X
        inst✝ : NormedSpace Real Y
        f : X → Y
        x : X
        H : IsConformalMap (fderiv Real f x)
        h : Not (DifferentiableAt Real f x)
        ⊢ ConformalAt f x
      -/
    · nontriviality X
      /-
        X : Type u_1
        Y : Type u_2
        inst✝³ : NormedAddCommGroup X
        inst✝² : NormedAddCommGroup Y
        inst✝¹ : NormedSpace Real X
        inst✝ : NormedSpace Real Y
        f : X → Y
        x : X
        H : IsConformalMap (fderiv Real f x)
        h : Not (DifferentiableAt Real f x)
        a✝ : Nontrivial X
        ⊢ ConformalAt f x
      -/
      exact absurd (fderiv_zero_of_not_differentiableAt h) H.ne_zero
      /-
        🎉 no goals
      -/


theorem differentiableAt {f : X → Y} {x : X} (h : ConformalAt f x) : DifferentiableAt ℝ f x :=
  let ⟨_, h₁, _⟩ := h
  h₁.differentiableAt


theorem congr {f g : X → Y} {x : X} {u : Set X} (hx : x ∈ u) (hu : IsOpen u) (hf : ConformalAt f x)
    (h : ∀ x : X, x ∈ u → g x = f x) : ConformalAt g x :=
  let ⟨f', hfderiv, hf'⟩ := hf
  ⟨f', hfderiv.congr_of_eventuallyEq ((hu.eventually_mem hx).mono h), hf'⟩


theorem comp {f : X → Y} {g : Y → Z} (x : X) (hg : ConformalAt g (f x)) (hf : ConformalAt f x) :
    ConformalAt (g ∘ f) x := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝⁵ : NormedAddCommGroup X
    inst✝⁴ : NormedAddCommGroup Y
    inst✝³ : NormedAddCommGroup Z
    inst✝² : NormedSpace Real X
    inst✝¹ : NormedSpace Real Y
    inst✝ : NormedSpace Real Z
    f : X → Y
    g : Y → Z
    x : X
    hg : ConformalAt g (f x)
    hf : ConformalAt f x
    ⊢ ConformalAt (Function.comp g f) x
  -/
  rcases hf with ⟨f', hf₁, cf⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝⁵ : NormedAddCommGroup X
    inst✝⁴ : NormedAddCommGroup Y
    inst✝³ : NormedAddCommGroup Z
    inst✝² : NormedSpace Real X
    inst✝¹ : NormedSpace Real Y
    inst✝ : NormedSpace Real Z
    f : X → Y
    g : Y → Z
    x : X
    hg : ConformalAt g (f x)
    f' : ContinuousLinearMap (RingHom.id Real) X Y
    hf₁ : HasFDerivAt f f' x
    cf : IsConformalMap f'
    ⊢ ConformalAt (Function.comp g f) x
  -/
  rcases hg with ⟨g', hg₁, cg⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝⁵ : NormedAddCommGroup X
    inst✝⁴ : NormedAddCommGroup Y
    inst✝³ : NormedAddCommGroup Z
    inst✝² : NormedSpace Real X
    inst✝¹ : NormedSpace Real Y
    inst✝ : NormedSpace Real Z
    f : X → Y
    g : Y → Z
    x : X
    f' : ContinuousLinearMap (RingHom.id Real) X Y
    hf₁ : HasFDerivAt f f' x
    cf : IsConformalMap f'
    g' : ContinuousLinearMap (RingHom.id Real) Y Z
    hg₁ : HasFDerivAt g g' (f x)
    cg : IsConformalMap g'
    ⊢ ConformalAt (Function.comp g f) x
  -/
  exact ⟨g'.comp f', hg₁.comp x hf₁, cg.comp cf⟩
  /-
    🎉 no goals
  -/


theorem const_smul {f : X → Y} {x : X} {c : ℝ} (hc : c ≠ 0) (hf : ConformalAt f x) :
    ConformalAt (c • f) x :=
  (conformalAt_const_smul hc <| f x).comp x hf


/-- A map `f` is conformal if it's conformal at every point. -/
def Conformal (f : X → Y) :=
  ∀ x : X, ConformalAt f x


theorem conformal_id : Conformal (id : X → X) := fun x => conformalAt_id x


theorem conformal_const_smul {c : ℝ} (h : c ≠ 0) : Conformal fun x : X => c • x := fun x =>
  conformalAt_const_smul h x


theorem conformalAt {f : X → Y} (h : Conformal f) (x : X) : ConformalAt f x :=
  h x


theorem differentiable {f : X → Y} (h : Conformal f) : Differentiable ℝ f := fun x =>
  (h x).differentiableAt


theorem comp {f : X → Y} {g : Y → Z} (hf : Conformal f) (hg : Conformal g) : Conformal (g ∘ f) :=
  fun x => (hg <| f x).comp x (hf x)


theorem const_smul {f : X → Y} (hf : Conformal f) {c : ℝ} (hc : c ≠ 0) : Conformal (c • f) :=
  fun x => (hf x).const_smul hc


