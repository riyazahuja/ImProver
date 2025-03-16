/-- Continuous path connecting two points `x` and `y` in a topological space -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
structure Path (x y : X) extends C(I, X) where
  /-- The start point of a `Path`. -/
  source' : toFun 0 = x
  /-- The end point of a `Path`. -/
  target' : toFun 1 = y


instance Path.funLike : FunLike (Path x y) I X where
  coe γ := ⇑γ.toContinuousMap
  coe_injective' γ₁ γ₂ h := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      x y z : X
      ι : Type u_3
      γ₁ γ₂ : Path x y
      h : Eq ((fun γ => ⇑γ.toContinuousMap) γ₁) ((fun γ => ⇑γ.toContinuousMap) γ₂)
      ⊢ Eq γ₁ γ₂
    -/
    simp only [DFunLike.coe_fn_eq] at h
    /-
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      x y z : X
      ι : Type u_3
      γ₁ γ₂ : Path x y
      h : Eq γ₁.toContinuousMap γ₂.toContinuousMap
      ⊢ Eq γ₁ γ₂
    -/
    cases γ₁; cases γ₂; congr
                        /-
                          🎉 no goals
                        -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added this instance so that we can use `FunLike.coe` for `CoeFun`
-- this also fixed very strange `simp` timeout issues

instance Path.continuousMapClass : ContinuousMapClass (Path x y) I X where
                                                           /-
                                                             X : Type u_1
                                                             Y : Type u_2
                                                             inst✝¹ : TopologicalSpace X
                                                             inst✝ : TopologicalSpace Y
                                                             x y z : X
                                                             ι : Type u_3
                                                             γ : Path x y
                                                             ⊢ Continuous ⇑γ.toContinuousMap
                                                           -/
  map_continuous γ := show Continuous γ.toContinuousMap by fun_prop
                                                           /-
                                                             🎉 no goals
                                                           -/


@[ext]
protected theorem Path.ext : ∀ {γ₁ γ₂ : Path x y}, (γ₁ : I → X) = γ₂ → γ₁ = γ₂ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    ⊢ ∀ {γ₁ γ₂ : Path x y}, Eq ⇑γ₁ ⇑γ₂ → Eq γ₁ γ₂
  -/
  rintro ⟨⟨x, h11⟩, h12, h13⟩ ⟨⟨x, h21⟩, h22, h23⟩ rfl
  /-
    case mk.mk.mk.mk
    X : Type u_1
    inst✝ : TopologicalSpace X
    x✝ y : X
    x : ↑unitInterval → X
    h11 : Continuous x
    h12 : Eq ({ toFun := x, continuous_toFun := h11 }.toFun 0) x✝
    h13 : Eq ({ toFun := x, continuous_toFun := h11 }.toFun 1) y
    h21 : Continuous ⇑{ toFun := x, continuous_toFun := h11, source' := h12, targe …
    h22 : Eq ({ toFun := ⇑{ toFun := x, continuous_toFun := h11, source' := h12, t …
    h23 : Eq ({ toFun := ⇑{ toFun := x, continuous_toFun := h11, source' := h12, t …
    ⊢ Eq { toFun := x, continuous_toFun := h11, source' := h12, target' := h13 } { …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_mk_mk (f : I → X) (h₁) (h₂ : f 0 = x) (h₃ : f 1 = y) :
    ⇑(mk ⟨f, h₁⟩ h₂ h₃ : Path x y) = f :=
  rfl
-- Porting note: the name `Path.coe_mk` better refers to a new lemma below


@[continuity]
protected theorem continuous : Continuous γ :=
  γ.continuous_toFun


@[simp]
protected theorem source : γ 0 = x :=
  γ.source'


@[simp]
protected theorem target : γ 1 = y :=
  γ.target'


/-- See Note [custom simps projection]. We need to specify this projection explicitly in this case,
because it is a composition of multiple projections. -/
def simps.apply : I → X :=
  γ


@[simp]
theorem coe_toContinuousMap : ⇑γ.toContinuousMap = γ :=
  rfl

-- Porting note: this is needed because of the `Path.continuousMapClass` instance

@[simp]
theorem coe_mk : ⇑(γ : C(I, X)) = γ :=
  rfl


/-- Any function `φ : Π (a : α), Path (x a) (y a)` can be seen as a function `α × I → X`. -/
instance hasUncurryPath {X α : Type*} [TopologicalSpace X] {x y : α → X} :
    HasUncurry (∀ a : α, Path (x a) (y a)) (α × I) X :=
  ⟨fun φ p => φ p.1 p.2⟩


/-- The constant path from a point to itself -/
@[refl, simps]
def refl (x : X) : Path x x where
  toFun _t := x
  continuous_toFun := continuous_const
  source' := rfl
  target' := rfl


@[simp]
                                                             /-
                                                               X : Type u_1
                                                               inst✝ : TopologicalSpace X
                                                               a : X
                                                               ⊢ Eq (Set.range ⇑(Path.refl a)) (Singleton.singleton a)
                                                             -/
theorem refl_range {a : X} : range (Path.refl a) = {a} := by simp [Path.refl, CoeFun.coe]
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- The reverse of a path from `x` to `y`, as a path from `y` to `x` -/
@[symm, simps]
def symm (γ : Path x y) : Path y x where
  toFun := γ ∘ σ
                         /-
                           X : Type u_1
                           Y : Type u_2
                           inst✝¹ : TopologicalSpace X
                           inst✝ : TopologicalSpace Y
                           x y z : X
                           ι : Type u_3
                           γ✝ γ : Path x y
                           ⊢ Continuous (Function.comp (⇑γ) unitInterval.symm)
                         -/
  continuous_toFun := by continuity
                         /-
                           🎉 no goals
                         -/
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ γ : Path x y
                  ⊢ Eq ({ toFun := Function.comp (⇑γ) unitInterval.symm, continuous_toFun := ⋯ } …
                -/
  source' := by simpa [-Path.target] using γ.target
                /-
                  🎉 no goals
                -/
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ γ : Path x y
                  ⊢ Eq ({ toFun := Function.comp (⇑γ) unitInterval.symm, continuous_toFun := ⋯ } …
                -/
  target' := by simpa [-Path.source] using γ.source
                /-
                  🎉 no goals
                -/


@[simp]
theorem symm_symm (γ : Path x y) : γ.symm.symm = γ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    ⊢ Eq γ.symm.symm γ
  -/
  ext t
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    t : ↑unitInterval
    ⊢ Eq (γ.symm.symm t) (γ t)
  -/
  show γ (σ (σ t)) = γ t
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    t : ↑unitInterval
    ⊢ Eq (γ (unitInterval.symm (unitInterval.symm t))) (γ t)
  -/
  rw [unitInterval.symm_symm]
  /-
    🎉 no goals
  -/


theorem symm_bijective : Function.Bijective (Path.symm : Path x y → Path y x) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


@[simp]
theorem refl_symm {a : X} : (Path.refl a).symm = Path.refl a := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a : X
    ⊢ Eq (Path.refl a).symm (Path.refl a)
  -/
  ext
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    a : X
    x✝ : ↑unitInterval
    ⊢ Eq ((Path.refl a).symm x✝) ((Path.refl a) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_range {a b : X} (γ : Path a b) : range γ.symm = range γ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    ⊢ Eq (Set.range ⇑γ.symm) (Set.range ⇑γ)
  -/
  ext x
  simp only [mem_range, Path.symm, DFunLike.coe, unitInterval.symm, SetCoe.exists, comp_apply,
    Subtype.coe_mk]
  /-
    case h
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    x : X
    ⊢ Iff (Exists fun x_1 => Exists fun h => Eq (γ.toFun ⟨HSub.hSub 1 x_1, ⋯⟩) x)  …
  -/
  constructor <;> rintro ⟨y, hy, hxy⟩ <;> refine ⟨1 - y, mem_iff_one_sub_mem.mp hy, ?_⟩ <;>
    /-
      case h.mp.intro.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      a b : X
      γ : Path a b
      x : X
      y : Real
      hy : Membership.mem unitInterval y
      hxy : Eq (γ.toFun ⟨HSub.hSub 1 y, ⋯⟩) x
      ⊢ Eq (γ.toFun ⟨HSub.hSub 1 y, ⋯⟩) x
    -/
    /-
      🎉 no goals
    -/
    convert hxy
  /-
    case h.e'_2.h.e'_6.h.e'_3
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    x : X
    y : Real
    hy : Membership.mem unitInterval y
    hxy : Eq (γ.toFun ⟨y, hy⟩) x
    ⊢ Eq (HSub.hSub 1 (HSub.hSub 1 y)) y
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The following instance defines the topology on the path space to be induced from the
compact-open topology on the space `C(I,X)` of continuous maps from `I` to `X`.
-/
instance topologicalSpace : TopologicalSpace (Path x y) :=
  TopologicalSpace.induced ((↑) : _ → C(I, X)) ContinuousMap.compactOpen


                                            /-
                                              X : Type u_1
                                              Y : Type u_2
                                              inst✝¹ : TopologicalSpace X
                                              inst✝ : TopologicalSpace Y
                                              x y z : X
                                              ι : Type u_3
                                              γ : Path x y
                                              ⊢ ∀ (g : Path x y), Eq ⇑↑g ⇑g
                                            -/
instance : ContinuousEval (Path x y) I X := .of_continuous_forget continuous_induced_dom
                                            /-
                                              🎉 no goals
                                            -/


@[deprecated (since := "2024-10-04")] protected alias continuous_eval := continuous_eval


@[deprecated Continuous.eval (since := "2024-10-04")]
theorem _root_.Continuous.path_eval {Y} [TopologicalSpace Y] {f : Y → Path x y} {g : Y → I}
    (hf : Continuous f) (hg : Continuous g) : Continuous fun y => f y (g y) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    x y : X
    Y : Type u_4
    inst✝ : TopologicalSpace Y
    f : Y → Path x y
    g : Y → ↑unitInterval
    hf : Continuous f
    hg : Continuous g
    ⊢ Continuous fun y_1 => (f y_1) (g y_1)
  -/
  continuity
  /-
    🎉 no goals
  -/


theorem continuous_uncurry_iff {Y} [TopologicalSpace Y] {g : Y → Path x y} :
    Continuous ↿g ↔ Continuous g :=
  Iff.symm <| continuous_induced_rng.trans
    ⟨fun h => continuous_uncurry_of_continuous ⟨_, h⟩,
                                                    /-
                                                      X : Type u_1
                                                      inst✝¹ : TopologicalSpace X
                                                      x y✝ : X
                                                      Y : Type u_4
                                                      inst✝ : TopologicalSpace Y
                                                      g : Y → Path x y✝
                                                      y : Y
                                                      ⊢ Continuous ⇑(g y)
                                                    -/
    continuous_of_continuous_uncurry (fun (y : Y) ↦ ContinuousMap.mk (g y))⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- A continuous map extending a path to `ℝ`, constant before `0` and after `1`. -/
def extend : ℝ → X :=
  IccExtend zero_le_one γ


/-- See Note [continuity lemma statement]. -/
theorem _root_.Continuous.path_extend {γ : Y → Path x y} {f : Y → ℝ} (hγ : Continuous ↿γ)
    (hf : Continuous f) : Continuous fun t => (γ t).extend (f t) :=
  Continuous.IccExtend hγ hf


/-- A useful special case of `Continuous.path_extend`. -/
@[continuity, fun_prop]
theorem continuous_extend : Continuous γ.extend :=
  γ.continuous.Icc_extend'


theorem _root_.Filter.Tendsto.path_extend
    {l r : Y → X} {y : Y} {l₁ : Filter ℝ} {l₂ : Filter X} {γ : ∀ y, Path (l y) (r y)}
    (hγ : Tendsto (↿γ) (𝓝 y ×ˢ l₁.map (projIcc 0 1 zero_le_one)) l₂) :
    Tendsto (↿fun x => (γ x).extend) (𝓝 y ×ˢ l₁) l₂ :=
  Filter.Tendsto.IccExtend _ hγ


theorem _root_.ContinuousAt.path_extend {g : Y → ℝ} {l r : Y → X} (γ : ∀ y, Path (l y) (r y))
    {y : Y} (hγ : ContinuousAt (↿γ) (y, projIcc 0 1 zero_le_one (g y))) (hg : ContinuousAt g y) :
    ContinuousAt (fun i => (γ i).extend (g i)) y :=
  hγ.IccExtend (fun x => γ x) hg


@[simp]
theorem extend_extends {a b : X} (γ : Path a b) {t : ℝ}
    (ht : t ∈ (Icc 0 1 : Set ℝ)) : γ.extend t = γ ⟨t, ht⟩ :=
  IccExtend_of_mem _ γ ht


                                           /-
                                             X : Type u_1
                                             inst✝ : TopologicalSpace X
                                             x y : X
                                             γ : Path x y
                                             ⊢ Eq (γ.extend 0) x
                                           -/
theorem extend_zero : γ.extend 0 = x := by simp
                                           /-
                                             🎉 no goals
                                           -/


                                          /-
                                            X : Type u_1
                                            inst✝ : TopologicalSpace X
                                            x y : X
                                            γ : Path x y
                                            ⊢ Eq (γ.extend 1) y
                                          -/
theorem extend_one : γ.extend 1 = y := by simp
                                          /-
                                            🎉 no goals
                                          -/


theorem extend_extends' {a b : X} (γ : Path a b) (t : (Icc 0 1 : Set ℝ)) : γ.extend t = γ t :=
  IccExtend_val _ γ t


@[simp]
theorem extend_range {a b : X} (γ : Path a b) :
    range γ.extend = range γ :=
  IccExtend_range _ γ


theorem extend_of_le_zero {a b : X} (γ : Path a b) {t : ℝ}
    (ht : t ≤ 0) : γ.extend t = a :=
  (IccExtend_of_le_left _ _ ht).trans γ.source


theorem extend_of_one_le {a b : X} (γ : Path a b) {t : ℝ}
    (ht : 1 ≤ t) : γ.extend t = b :=
  (IccExtend_of_right_le _ _ ht).trans γ.target


@[simp]
theorem refl_extend {a : X} : (Path.refl a).extend = fun _ => a :=
  rfl


/-- The path obtained from a map defined on `ℝ` by restriction to the unit interval. -/
def ofLine {f : ℝ → X} (hf : ContinuousOn f I) (h₀ : f 0 = x) (h₁ : f 1 = y) : Path x y where
  toFun := f ∘ ((↑) : unitInterval → ℝ)
  continuous_toFun := hf.comp_continuous continuous_subtype_val Subtype.prop
  source' := h₀
  target' := h₁


theorem ofLine_mem {f : ℝ → X} (hf : ContinuousOn f I) (h₀ : f 0 = x) (h₁ : f 1 = y) :
    ∀ t, ofLine hf h₀ h₁ t ∈ f '' I := fun ⟨t, t_in⟩ => ⟨t, t_in, rfl⟩


/-- Concatenation of two paths from `x` to `y` and from `y` to `z`, putting the first
path on `[0, 1/2]` and the second one on `[1/2, 1]`. -/
@[trans]
def trans (γ : Path x y) (γ' : Path y z) : Path x z where
  toFun := (fun t : ℝ => if t ≤ 1 / 2 then γ.extend (2 * t) else γ'.extend (2 * t - 1)) ∘ (↑)
  continuous_toFun := by
    refine
      (Continuous.if_le ?_ ?_ continuous_id continuous_const (by norm_num)).comp
        continuous_subtype_val <;>
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      x y z : X
      ι : Type u_3
      γ✝ γ : Path x y
      γ' : Path y z
      ⊢ Continuous fun t => γ.extend (HMul.hMul 2 t)
    -/
    /-
      🎉 no goals
    -/
    fun_prop
    /-
      🎉 no goals
    -/
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ γ : Path x y
                  γ' : Path y z
                  ⊢ Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ.extend (HMul …
                -/
  source' := by norm_num
                /-
                  🎉 no goals
                -/
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ γ : Path x y
                  γ' : Path y z
                  ⊢ Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ.extend (HMul …
                -/
  target' := by norm_num
                /-
                  🎉 no goals
                -/


theorem trans_apply (γ : Path x y) (γ' : Path y z) (t : I) :
    (γ.trans γ') t =
      if h : (t : ℝ) ≤ 1 / 2 then γ ⟨2 * t, (mul_pos_mem_iff zero_lt_two).2 ⟨t.2.1, h⟩⟩
      else γ' ⟨2 * t - 1, two_mul_sub_one_mem_iff.2 ⟨(not_le.1 h).le, t.2.2⟩⟩ :=
                        /-
                          X : Type u_1
                          inst✝ : TopologicalSpace X
                          x y z : X
                          γ : Path x y
                          γ' : Path y z
                          t : ↑unitInterval
                          ⊢ Eq (ite (LE.le (↑t) (1 / 2)) (γ.extend (HMul.hMul 2 ↑t)) (γ'.extend (HSub.hS …
                        -/
                                      /-
                                        🎉 no goals
                                      -/
  show ite _ _ _ = _ by split_ifs <;> rw [extend_extends]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem trans_symm (γ : Path x y) (γ' : Path y z) : (γ.trans γ').symm = γ'.symm.trans γ.symm := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y z : X
    γ : Path x y
    γ' : Path y z
    ⊢ Eq (γ.trans γ').symm (γ'.symm.trans γ.symm)
  -/
  ext t
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y z : X
    γ : Path x y
    γ' : Path y z
    t : ↑unitInterval
    ⊢ Eq ((γ.trans γ').symm t) ((γ'.symm.trans γ.symm) t)
  -/
  simp only [trans_apply, ← one_div, symm_apply, not_le, Function.comp_apply]
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y z : X
    γ : Path x y
    γ' : Path y z
    t : ↑unitInterval
    ⊢ Eq (dite (LE.le (↑(unitInterval.symm t)) (1 / 2)) (fun h => γ ⟨HMul.hMul 2 ↑ …
  -/
  split_ifs with h h₁ h₂ <;> rw [coe_symm_eq] at h
    /-
      case pos
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y z : X
      γ : Path x y
      γ' : Path y z
      t : ↑unitInterval
      h✝ : LE.le (↑(unitInterval.symm t)) (1 / 2)
      h : LE.le (HSub.hSub 1 ↑t) (1 / 2)
      h₁ : LE.le (↑t) (1 / 2)
      ⊢ Eq (γ ⟨HMul.hMul 2 ↑(unitInterval.symm t), ⋯⟩) (γ' (unitInterval.symm ⟨HMul. …
    -/
  · have ht : (t : ℝ) = 1 / 2 := by linarith
    /-
      case pos
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y z : X
      γ : Path x y
      γ' : Path y z
      t : ↑unitInterval
      h✝ : LE.le (↑(unitInterval.symm t)) (1 / 2)
      h : LE.le (HSub.hSub 1 ↑t) (1 / 2)
      h₁ : LE.le (↑t) (1 / 2)
      ht : Eq (↑t) (1 / 2)
      ⊢ Eq (γ ⟨HMul.hMul 2 ↑(unitInterval.symm t), ⋯⟩) (γ' (unitInterval.symm ⟨HMul. …
    -/
    norm_num [ht]
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y z : X
      γ : Path x y
      γ' : Path y z
      t : ↑unitInterval
      h✝ : LE.le (↑(unitInterval.symm t)) (1 / 2)
      h : LE.le (HSub.hSub 1 ↑t) (1 / 2)
      h₁ : Not (LE.le (↑t) (1 / 2))
      ⊢ Eq (γ ⟨HMul.hMul 2 ↑(unitInterval.symm t), ⋯⟩) (γ (unitInterval.symm ⟨HSub.h …
    -/
  · refine congr_arg _ (Subtype.ext ?_)
    /-
      case neg
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y z : X
      γ : Path x y
      γ' : Path y z
      t : ↑unitInterval
      h✝ : LE.le (↑(unitInterval.symm t)) (1 / 2)
      h : LE.le (HSub.hSub 1 ↑t) (1 / 2)
      h₁ : Not (LE.le (↑t) (1 / 2))
      ⊢ Eq ↑⟨HMul.hMul 2 ↑(unitInterval.symm t), ⋯⟩ ↑(unitInterval.symm ⟨HSub.hSub ( …
    -/
    norm_num [sub_sub_eq_add_sub, mul_sub]
    /-
      🎉 no goals
    -/
    /-
      case pos
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y z : X
      γ : Path x y
      γ' : Path y z
      t : ↑unitInterval
      h✝ : Not (LE.le (↑(unitInterval.symm t)) (1 / 2))
      h : Not (LE.le (HSub.hSub 1 ↑t) (1 / 2))
      h₂ : LE.le (↑t) (1 / 2)
      ⊢ Eq (γ' ⟨HSub.hSub (HMul.hMul 2 ↑(unitInterval.symm t)) 1, ⋯⟩) (γ' (unitInter …
    -/
  · refine congr_arg _ (Subtype.ext ?_)
    /-
      case pos
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y z : X
      γ : Path x y
      γ' : Path y z
      t : ↑unitInterval
      h✝ : Not (LE.le (↑(unitInterval.symm t)) (1 / 2))
      h : Not (LE.le (HSub.hSub 1 ↑t) (1 / 2))
      h₂ : LE.le (↑t) (1 / 2)
      ⊢ Eq ↑⟨HSub.hSub (HMul.hMul 2 ↑(unitInterval.symm t)) 1, ⋯⟩ ↑(unitInterval.sym …
    -/
    norm_num [mul_sub, h]
    /-
      case pos
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y z : X
      γ : Path x y
      γ' : Path y z
      t : ↑unitInterval
      h✝ : Not (LE.le (↑(unitInterval.symm t)) (1 / 2))
      h : Not (LE.le (HSub.hSub 1 ↑t) (1 / 2))
      h₂ : LE.le (↑t) (1 / 2)
      ⊢ Eq (HSub.hSub (HSub.hSub 2 (HMul.hMul 2 ↑t)) 1) (HSub.hSub 1 (HMul.hMul 2 ↑t))
    -/
    ring -- TODO norm_num should really do this
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y z : X
      γ : Path x y
      γ' : Path y z
      t : ↑unitInterval
      h✝ : Not (LE.le (↑(unitInterval.symm t)) (1 / 2))
      h : Not (LE.le (HSub.hSub 1 ↑t) (1 / 2))
      h₂ : Not (LE.le (↑t) (1 / 2))
      ⊢ Eq (γ' ⟨HSub.hSub (HMul.hMul 2 ↑(unitInterval.symm t)) 1, ⋯⟩) (γ (unitInterv …
    -/
  · exfalso
    /-
      case neg
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y z : X
      γ : Path x y
      γ' : Path y z
      t : ↑unitInterval
      h✝ : Not (LE.le (↑(unitInterval.symm t)) (1 / 2))
      h : Not (LE.le (HSub.hSub 1 ↑t) (1 / 2))
      h₂ : Not (LE.le (↑t) (1 / 2))
      ⊢ False
    -/
    linarith
    /-
      🎉 no goals
    -/


@[simp]
theorem refl_trans_refl {a : X} :
    (Path.refl a).trans (Path.refl a) = Path.refl a := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a : X
    ⊢ Eq ((Path.refl a).trans (Path.refl a)) (Path.refl a)
  -/
  ext
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    a : X
    x✝ : ↑unitInterval
    ⊢ Eq (((Path.refl a).trans (Path.refl a)) x✝) ((Path.refl a) x✝)
  -/
  simp only [Path.trans, ite_self, one_div, Path.refl_extend]
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    a : X
    x✝ : ↑unitInterval
    ⊢ Eq ({ toFun := Function.comp (fun t => a) Subtype.val, continuous_toFun := ⋯ …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem trans_range {a b c : X} (γ₁ : Path a b) (γ₂ : Path b c) :
    range (γ₁.trans γ₂) = range γ₁ ∪ range γ₂ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b c : X
    γ₁ : Path a b
    γ₂ : Path b c
    ⊢ Eq (Set.range ⇑(γ₁.trans γ₂)) (Union.union (Set.range ⇑γ₁) (Set.range ⇑γ₂))
  -/
  rw [Path.trans]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b c : X
    γ₁ : Path a b
    γ₂ : Path b c
    ⊢ Eq (Set.range ⇑{ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁. …
  -/
  apply eq_of_subset_of_subset
    /-
      case a
      X : Type u_1
      inst✝ : TopologicalSpace X
      a b c : X
      γ₁ : Path a b
      γ₂ : Path b c
      ⊢ HasSubset.Subset (Set.range ⇑{ toFun := Function.comp (fun t => ite (LE.le t …
    -/
  · rintro x ⟨⟨t, ht0, ht1⟩, hxt⟩
    /-
      case a.intro.mk.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      a b c : X
      γ₁ : Path a b
      γ₂ : Path b c
      x : X
      t : Real
      ht0 : LE.le 0 t
      ht1 : LE.le t 1
      hxt : Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend  …
      ⊢ Membership.mem (Union.union (Set.range ⇑γ₁) (Set.range ⇑γ₂)) x
    -/
    by_cases h : t ≤ 1 / 2
      /-
        case pos
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend  …
        h : LE.le t (1 / 2)
        ⊢ Membership.mem (Union.union (Set.range ⇑γ₁) (Set.range ⇑γ₂)) x
      -/
    · left
      /-
        case pos.h
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend  …
        h : LE.le t (1 / 2)
        ⊢ Membership.mem (Set.range ⇑γ₁) x
      -/
      use ⟨2 * t, ⟨by linarith, by linarith⟩⟩
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend  …
        h : LE.le t (1 / 2)
        ⊢ Eq (γ₁ ⟨HMul.hMul 2 t, ⋯⟩) x
      -/
      rw [← γ₁.extend_extends]
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend  …
        h : LE.le t (1 / 2)
        ⊢ Eq (γ₁.extend (HMul.hMul 2 t)) x
      -/
      rwa [coe_mk_mk, Function.comp_apply, if_pos h] at hxt
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend  …
        h : Not (LE.le t (1 / 2))
        ⊢ Membership.mem (Union.union (Set.range ⇑γ₁) (Set.range ⇑γ₂)) x
      -/
    · right
      /-
        case neg.h
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend  …
        h : Not (LE.le t (1 / 2))
        ⊢ Membership.mem (Set.range ⇑γ₂) x
      -/
      use ⟨2 * t - 1, ⟨by linarith, by linarith⟩⟩
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend  …
        h : Not (LE.le t (1 / 2))
        ⊢ Eq (γ₂ ⟨HSub.hSub (HMul.hMul 2 t) 1, ⋯⟩) x
      -/
      rw [← γ₂.extend_extends]
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend  …
        h : Not (LE.le t (1 / 2))
        ⊢ Eq (γ₂.extend (HSub.hSub (HMul.hMul 2 t) 1)) x
      -/
      rwa [coe_mk_mk, Function.comp_apply, if_neg h] at hxt
      /-
        🎉 no goals
      -/
    /-
      case a
      X : Type u_1
      inst✝ : TopologicalSpace X
      a b c : X
      γ₁ : Path a b
      γ₂ : Path b c
      ⊢ HasSubset.Subset (Union.union (Set.range ⇑γ₁) (Set.range ⇑γ₂)) (Set.range ⇑{ …
    -/
  · rintro x (⟨⟨t, ht0, ht1⟩, hxt⟩ | ⟨⟨t, ht0, ht1⟩, hxt⟩)
      /-
        case a.inl.intro.mk.intro
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq (γ₁ ⟨t, ⋯⟩) x
        ⊢ Membership.mem (Set.range ⇑{ toFun := Function.comp (fun t => ite (LE.le t ( …
      -/
    · use ⟨t / 2, ⟨by linarith, by linarith⟩⟩
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq (γ₁ ⟨t, ⋯⟩) x
        ⊢ Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend (HMu …
      -/
      have : t / 2 ≤ 1 / 2 := (div_le_div_iff_of_pos_right (zero_lt_two : (0 : ℝ) < 2)).mpr ht1
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq (γ₁ ⟨t, ⋯⟩) x
        this : LE.le (HDiv.hDiv t 2) (1 / 2)
        ⊢ Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend (HMu …
      -/
      rw [coe_mk_mk, Function.comp_apply, if_pos this, Subtype.coe_mk]
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq (γ₁ ⟨t, ⋯⟩) x
        this : LE.le (HDiv.hDiv t 2) (1 / 2)
        ⊢ Eq (γ₁.extend (HMul.hMul 2 (HDiv.hDiv t 2))) x
      -/
      ring_nf
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq (γ₁ ⟨t, ⋯⟩) x
        this : LE.le (HDiv.hDiv t 2) (1 / 2)
        ⊢ Eq (γ₁.extend t) x
      -/
      rwa [γ₁.extend_extends]
      /-
        🎉 no goals
      -/
      /-
        case a.inr.intro.mk.intro
        X : Type u_1
        inst✝ : TopologicalSpace X
        a b c : X
        γ₁ : Path a b
        γ₂ : Path b c
        x : X
        t : Real
        ht0 : LE.le 0 t
        ht1 : LE.le t 1
        hxt : Eq (γ₂ ⟨t, ⋯⟩) x
        ⊢ Membership.mem (Set.range ⇑{ toFun := Function.comp (fun t => ite (LE.le t ( …
      -/
    · by_cases h : t = 0
        /-
          case pos
          X : Type u_1
          inst✝ : TopologicalSpace X
          a b c : X
          γ₁ : Path a b
          γ₂ : Path b c
          x : X
          t : Real
          ht0 : LE.le 0 t
          ht1 : LE.le t 1
          hxt : Eq (γ₂ ⟨t, ⋯⟩) x
          h : Eq t 0
          ⊢ Membership.mem (Set.range ⇑{ toFun := Function.comp (fun t => ite (LE.le t ( …
        -/
      · use ⟨1 / 2, ⟨by linarith, by linarith⟩⟩
        rw [coe_mk_mk, Function.comp_apply, if_pos le_rfl, Subtype.coe_mk,
          mul_one_div_cancel (two_ne_zero' ℝ)]
        /-
          case h
          X : Type u_1
          inst✝ : TopologicalSpace X
          a b c : X
          γ₁ : Path a b
          γ₂ : Path b c
          x : X
          t : Real
          ht0 : LE.le 0 t
          ht1 : LE.le t 1
          hxt : Eq (γ₂ ⟨t, ⋯⟩) x
          h : Eq t 0
          ⊢ Eq (γ₁.extend 1) x
        -/
        rw [γ₁.extend_one]
        /-
          case h
          X : Type u_1
          inst✝ : TopologicalSpace X
          a b c : X
          γ₁ : Path a b
          γ₂ : Path b c
          x : X
          t : Real
          ht0 : LE.le 0 t
          ht1 : LE.le t 1
          hxt : Eq (γ₂ ⟨t, ⋯⟩) x
          h : Eq t 0
          ⊢ Eq b x
        -/
        rwa [← γ₂.extend_extends, h, γ₂.extend_zero] at hxt
        /-
          🎉 no goals
        -/
        /-
          case neg
          X : Type u_1
          inst✝ : TopologicalSpace X
          a b c : X
          γ₁ : Path a b
          γ₂ : Path b c
          x : X
          t : Real
          ht0 : LE.le 0 t
          ht1 : LE.le t 1
          hxt : Eq (γ₂ ⟨t, ⋯⟩) x
          h : Not (Eq t 0)
          ⊢ Membership.mem (Set.range ⇑{ toFun := Function.comp (fun t => ite (LE.le t ( …
        -/
      · use ⟨(t + 1) / 2, ⟨by linarith, by linarith⟩⟩
        /-
          case h
          X : Type u_1
          inst✝ : TopologicalSpace X
          a b c : X
          γ₁ : Path a b
          γ₂ : Path b c
          x : X
          t : Real
          ht0 : LE.le 0 t
          ht1 : LE.le t 1
          hxt : Eq (γ₂ ⟨t, ⋯⟩) x
          h : Not (Eq t 0)
          ⊢ Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend (HMu …
        -/
        replace h : t ≠ 0 := h
        /-
          case h
          X : Type u_1
          inst✝ : TopologicalSpace X
          a b c : X
          γ₁ : Path a b
          γ₂ : Path b c
          x : X
          t : Real
          ht0 : LE.le 0 t
          ht1 : LE.le t 1
          hxt : Eq (γ₂ ⟨t, ⋯⟩) x
          h : Ne t 0
          ⊢ Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend (HMu …
        -/
        have ht0 := lt_of_le_of_ne ht0 h.symm
        have : ¬(t + 1) / 2 ≤ 1 / 2 := by
          rw [not_le]
          linarith
        /-
          case h
          X : Type u_1
          inst✝ : TopologicalSpace X
          a b c : X
          γ₁ : Path a b
          γ₂ : Path b c
          x : X
          t : Real
          ht0✝ : LE.le 0 t
          ht1 : LE.le t 1
          hxt : Eq (γ₂ ⟨t, ⋯⟩) x
          h : Ne t 0
          ht0 : LT.lt 0 t
          this : Not (LE.le (HDiv.hDiv (HAdd.hAdd t 1) 2) (1 / 2))
          ⊢ Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) (γ₁.extend (HMu …
        -/
        rw [coe_mk_mk, Function.comp_apply, Subtype.coe_mk, if_neg this]
        /-
          case h
          X : Type u_1
          inst✝ : TopologicalSpace X
          a b c : X
          γ₁ : Path a b
          γ₂ : Path b c
          x : X
          t : Real
          ht0✝ : LE.le 0 t
          ht1 : LE.le t 1
          hxt : Eq (γ₂ ⟨t, ⋯⟩) x
          h : Ne t 0
          ht0 : LT.lt 0 t
          this : Not (LE.le (HDiv.hDiv (HAdd.hAdd t 1) 2) (1 / 2))
          ⊢ Eq (γ₂.extend (HSub.hSub (HMul.hMul 2 (HDiv.hDiv (HAdd.hAdd t 1) 2)) 1)) x
        -/
        ring_nf
        /-
          case h
          X : Type u_1
          inst✝ : TopologicalSpace X
          a b c : X
          γ₁ : Path a b
          γ₂ : Path b c
          x : X
          t : Real
          ht0✝ : LE.le 0 t
          ht1 : LE.le t 1
          hxt : Eq (γ₂ ⟨t, ⋯⟩) x
          h : Ne t 0
          ht0 : LT.lt 0 t
          this : Not (LE.le (HDiv.hDiv (HAdd.hAdd t 1) 2) (1 / 2))
          ⊢ Eq (γ₂.extend t) x
        -/
        rwa [γ₂.extend_extends]
        /-
          🎉 no goals
        -/


/-- Image of a path from `x` to `y` by a map which is continuous on the path. -/
def map' (γ : Path x y) {f : X → Y} (h : ContinuousOn f (range γ)) : Path (f x) (f y) where
  toFun := f ∘ γ
  continuous_toFun := h.comp_continuous γ.continuous (fun x ↦ mem_range_self x)
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ γ : Path x y
                  f : X → Y
                  h : ContinuousOn f (Set.range ⇑γ)
                  ⊢ Eq ({ toFun := Function.comp f ⇑γ, continuous_toFun := ⋯ }.toFun 0) (f x)
                -/
  source' := by simp
                /-
                  🎉 no goals
                -/
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ γ : Path x y
                  f : X → Y
                  h : ContinuousOn f (Set.range ⇑γ)
                  ⊢ Eq ({ toFun := Function.comp f ⇑γ, continuous_toFun := ⋯ }.toFun 1) (f y)
                -/
  target' := by simp
                /-
                  🎉 no goals
                -/


/-- Image of a path from `x` to `y` by a continuous map -/
def map (γ : Path x y) {f : X → Y} (h : Continuous f) :
    Path (f x) (f y) := γ.map' h.continuousOn


@[simp]
theorem map_coe (γ : Path x y) {f : X → Y} (h : Continuous f) :
    (γ.map h : I → Y) = f ∘ γ := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y : X
    γ : Path x y
    f : X → Y
    h : Continuous f
    ⊢ Eq (⇑(γ.map h)) (Function.comp f ⇑γ)
  -/
  ext t
  /-
    case h
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y : X
    γ : Path x y
    f : X → Y
    h : Continuous f
    t : ↑unitInterval
    ⊢ Eq ((γ.map h) t) (Function.comp f (⇑γ) t)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem map_symm (γ : Path x y) {f : X → Y} (h : Continuous f) :
    (γ.map h).symm = γ.symm.map h :=
  rfl


@[simp]
theorem map_trans (γ : Path x y) (γ' : Path y z) {f : X → Y}
    (h : Continuous f) : (γ.trans γ').map h = (γ.map h).trans (γ'.map h) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y z : X
    γ : Path x y
    γ' : Path y z
    f : X → Y
    h : Continuous f
    ⊢ Eq ((γ.trans γ').map h) ((γ.map h).trans (γ'.map h))
  -/
  ext t
  /-
    case a.h
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y z : X
    γ : Path x y
    γ' : Path y z
    f : X → Y
    h : Continuous f
    t : ↑unitInterval
    ⊢ Eq (((γ.trans γ').map h) t) (((γ.map h).trans (γ'.map h)) t)
  -/
  rw [trans_apply, map_coe, Function.comp_apply, trans_apply]
  /-
    case a.h
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y z : X
    γ : Path x y
    γ' : Path y z
    f : X → Y
    h : Continuous f
    t : ↑unitInterval
    ⊢ Eq (f (dite (LE.le (↑t) (1 / 2)) (fun h => γ ⟨HMul.hMul 2 ↑t, ⋯⟩) fun h => γ …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> rfl
                /-
                  🎉 no goals
                -/


@[simp]
theorem map_id (γ : Path x y) : γ.map continuous_id = γ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    ⊢ Eq (γ.map ⋯) γ
  -/
  ext
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    x✝ : ↑unitInterval
    ⊢ Eq ((γ.map ⋯) x✝) (γ x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem map_map (γ : Path x y) {Z : Type*} [TopologicalSpace Z]
    {f : X → Y} (hf : Continuous f) {g : Y → Z} (hg : Continuous g) :
    (γ.map hf).map hg = γ.map (hg.comp hf) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    x y : X
    γ : Path x y
    Z : Type u_4
    inst✝ : TopologicalSpace Z
    f : X → Y
    hf : Continuous f
    g : Y → Z
    hg : Continuous g
    ⊢ Eq ((γ.map hf).map hg) (γ.map ⋯)
  -/
  ext
  /-
    case a.h
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    x y : X
    γ : Path x y
    Z : Type u_4
    inst✝ : TopologicalSpace Z
    f : X → Y
    hf : Continuous f
    g : Y → Z
    hg : Continuous g
    x✝ : ↑unitInterval
    ⊢ Eq (((γ.map hf).map hg) x✝) ((γ.map ⋯) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Casting a path from `x` to `y` to a path from `x'` to `y'` when `x' = x` and `y' = y` -/
def cast (γ : Path x y) {x' y'} (hx : x' = x) (hy : y' = y) : Path x' y' where
  toFun := γ
  continuous_toFun := γ.continuous
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ γ : Path x y
                  x' y' : X
                  hx : Eq x' x
                  hy : Eq y' y
                  ⊢ Eq ({ toFun := ⇑γ, continuous_toFun := ⋯ }.toFun 0) x'
                -/
  source' := by simp [hx]
                /-
                  🎉 no goals
                -/
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ γ : Path x y
                  x' y' : X
                  hx : Eq x' x
                  hy : Eq y' y
                  ⊢ Eq ({ toFun := ⇑γ, continuous_toFun := ⋯ }.toFun 1) y'
                -/
  target' := by simp [hy]
                /-
                  🎉 no goals
                -/


@[simp]
theorem symm_cast {a₁ a₂ b₁ b₂ : X} (γ : Path a₂ b₂) (ha : a₁ = a₂) (hb : b₁ = b₂) :
    (γ.cast ha hb).symm = γ.symm.cast hb ha :=
  rfl


@[simp]
theorem trans_cast {a₁ a₂ b₁ b₂ c₁ c₂ : X} (γ : Path a₂ b₂)
    (γ' : Path b₂ c₂) (ha : a₁ = a₂) (hb : b₁ = b₂) (hc : c₁ = c₂) :
    (γ.cast ha hb).trans (γ'.cast hb hc) = (γ.trans γ').cast ha hc :=
  rfl


@[simp]
theorem cast_coe (γ : Path x y) {x' y'} (hx : x' = x) (hy : y' = y) : (γ.cast hx hy : I → X) = γ :=
  rfl


@[continuity, fun_prop]
theorem symm_continuous_family {ι : Type*} [TopologicalSpace ι]
    {a b : ι → X} (γ : ∀ t : ι, Path (a t) (b t)) (h : Continuous ↿γ) :
    Continuous ↿fun t => (γ t).symm :=
  h.comp (continuous_id.prodMap continuous_symm)


@[continuity]
theorem continuous_symm : Continuous (symm : Path x y → Path y x) :=
  continuous_uncurry_iff.mp <| symm_continuous_family _ (continuous_fst.eval continuous_snd)


@[continuity]
theorem continuous_uncurry_extend_of_continuous_family {ι : Type*} [TopologicalSpace ι]
    {a b : ι → X} (γ : ∀ t : ι, Path (a t) (b t)) (h : Continuous ↿γ) :
    Continuous ↿fun t => (γ t).extend := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    ι : Type u_4
    inst✝ : TopologicalSpace ι
    a b : ι → X
    γ : (t : ι) → Path (a t) (b t)
    h : Continuous (Function.HasUncurry.uncurry γ)
    ⊢ Continuous (Function.HasUncurry.uncurry fun t => (γ t).extend)
  -/
  apply h.comp (continuous_id.prodMap continuous_projIcc)
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    ι : Type u_4
    inst✝ : TopologicalSpace ι
    a b : ι → X
    γ : (t : ι) → Path (a t) (b t)
    h : Continuous (Function.HasUncurry.uncurry γ)
    ⊢ LE.le 0 1
  -/
  exact zero_le_one
  /-
    🎉 no goals
  -/


@[continuity]
theorem trans_continuous_family {ι : Type*} [TopologicalSpace ι]
    {a b c : ι → X} (γ₁ : ∀ t : ι, Path (a t) (b t)) (h₁ : Continuous ↿γ₁)
    (γ₂ : ∀ t : ι, Path (b t) (c t)) (h₂ : Continuous ↿γ₂) :
    Continuous ↿fun t => (γ₁ t).trans (γ₂ t) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    ι : Type u_4
    inst✝ : TopologicalSpace ι
    a b c : ι → X
    γ₁ : (t : ι) → Path (a t) (b t)
    h₁ : Continuous (Function.HasUncurry.uncurry γ₁)
    γ₂ : (t : ι) → Path (b t) (c t)
    h₂ : Continuous (Function.HasUncurry.uncurry γ₂)
    ⊢ Continuous (Function.HasUncurry.uncurry fun t => (γ₁ t).trans (γ₂ t))
  -/
  have h₁' := Path.continuous_uncurry_extend_of_continuous_family γ₁ h₁
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    ι : Type u_4
    inst✝ : TopologicalSpace ι
    a b c : ι → X
    γ₁ : (t : ι) → Path (a t) (b t)
    h₁ : Continuous (Function.HasUncurry.uncurry γ₁)
    γ₂ : (t : ι) → Path (b t) (c t)
    h₂ : Continuous (Function.HasUncurry.uncurry γ₂)
    h₁' : Continuous (Function.HasUncurry.uncurry fun t => (γ₁ t).extend)
    ⊢ Continuous (Function.HasUncurry.uncurry fun t => (γ₁ t).trans (γ₂ t))
  -/
  have h₂' := Path.continuous_uncurry_extend_of_continuous_family γ₂ h₂
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    ι : Type u_4
    inst✝ : TopologicalSpace ι
    a b c : ι → X
    γ₁ : (t : ι) → Path (a t) (b t)
    h₁ : Continuous (Function.HasUncurry.uncurry γ₁)
    γ₂ : (t : ι) → Path (b t) (c t)
    h₂ : Continuous (Function.HasUncurry.uncurry γ₂)
    h₁' : Continuous (Function.HasUncurry.uncurry fun t => (γ₁ t).extend)
    h₂' : Continuous (Function.HasUncurry.uncurry fun t => (γ₂ t).extend)
    ⊢ Continuous (Function.HasUncurry.uncurry fun t => (γ₁ t).trans (γ₂ t))
  -/
  simp only [HasUncurry.uncurry, CoeFun.coe, Path.trans, (· ∘ ·)]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    ι : Type u_4
    inst✝ : TopologicalSpace ι
    a b c : ι → X
    γ₁ : (t : ι) → Path (a t) (b t)
    h₁ : Continuous (Function.HasUncurry.uncurry γ₁)
    γ₂ : (t : ι) → Path (b t) (c t)
    h₂ : Continuous (Function.HasUncurry.uncurry γ₂)
    h₁' : Continuous (Function.HasUncurry.uncurry fun t => (γ₁ t).extend)
    h₂' : Continuous (Function.HasUncurry.uncurry fun t => (γ₂ t).extend)
    ⊢ Continuous fun p => { toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) …
  -/
  refine Continuous.if_le ?_ ?_ (continuous_subtype_val.comp continuous_snd) continuous_const ?_
  · change
      Continuous ((fun p : ι × ℝ => (γ₁ p.1).extend p.2) ∘ Prod.map id (fun x => 2 * x : I → ℝ))
    /-
      case refine_1
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      ι : Type u_4
      inst✝ : TopologicalSpace ι
      a b c : ι → X
      γ₁ : (t : ι) → Path (a t) (b t)
      h₁ : Continuous (Function.HasUncurry.uncurry γ₁)
      γ₂ : (t : ι) → Path (b t) (c t)
      h₂ : Continuous (Function.HasUncurry.uncurry γ₂)
      h₁' : Continuous (Function.HasUncurry.uncurry fun t => (γ₁ t).extend)
      h₂' : Continuous (Function.HasUncurry.uncurry fun t => (γ₂ t).extend)
      ⊢ Continuous (Function.comp (fun p => (γ₁ p.1).extend p.2) (Prod.map id fun x  …
    -/
    exact h₁'.comp (continuous_id.prodMap <| continuous_const.mul continuous_subtype_val)
    /-
      🎉 no goals
    -/
  · change
      Continuous ((fun p : ι × ℝ => (γ₂ p.1).extend p.2) ∘ Prod.map id (fun x => 2 * x - 1 : I → ℝ))
    exact
      h₂'.comp
        (continuous_id.prodMap <|
          (continuous_const.mul continuous_subtype_val).sub continuous_const)
    /-
      case refine_3
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      ι : Type u_4
      inst✝ : TopologicalSpace ι
      a b c : ι → X
      γ₁ : (t : ι) → Path (a t) (b t)
      h₁ : Continuous (Function.HasUncurry.uncurry γ₁)
      γ₂ : (t : ι) → Path (b t) (c t)
      h₂ : Continuous (Function.HasUncurry.uncurry γ₂)
      h₁' : Continuous (Function.HasUncurry.uncurry fun t => (γ₁ t).extend)
      h₂' : Continuous (Function.HasUncurry.uncurry fun t => (γ₂ t).extend)
      ⊢ ∀ (x : Prod ι ↑unitInterval), Eq (↑x.2) (1 / 2) → Eq ((γ₁ x.1).extend (HMul. …
    -/
  · rintro st hst
    /-
      case refine_3
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      ι : Type u_4
      inst✝ : TopologicalSpace ι
      a b c : ι → X
      γ₁ : (t : ι) → Path (a t) (b t)
      h₁ : Continuous (Function.HasUncurry.uncurry γ₁)
      γ₂ : (t : ι) → Path (b t) (c t)
      h₂ : Continuous (Function.HasUncurry.uncurry γ₂)
      h₁' : Continuous (Function.HasUncurry.uncurry fun t => (γ₁ t).extend)
      h₂' : Continuous (Function.HasUncurry.uncurry fun t => (γ₂ t).extend)
      st : Prod ι ↑unitInterval
      hst : Eq (↑st.2) (1 / 2)
      ⊢ Eq ((γ₁ st.1).extend (HMul.hMul 2 ↑st.2)) ((γ₂ st.1).extend (HSub.hSub (HMul …
    -/
    simp [hst, mul_inv_cancel₀ (two_ne_zero' ℝ)]
    /-
      🎉 no goals
    -/


@[continuity]
theorem _root_.Continuous.path_trans {f : Y → Path x y} {g : Y → Path y z} :
    Continuous f → Continuous g → Continuous fun t => (f t).trans (g t) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y z : X
    f : Y → Path x y
    g : Y → Path y z
    ⊢ Continuous f → Continuous g → Continuous fun t => (f t).trans (g t)
  -/
  intro hf hg
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y z : X
    f : Y → Path x y
    g : Y → Path y z
    hf : Continuous f
    hg : Continuous g
    ⊢ Continuous fun t => (f t).trans (g t)
  -/
  apply continuous_uncurry_iff.mp
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y z : X
    f : Y → Path x y
    g : Y → Path y z
    hf : Continuous f
    hg : Continuous g
    ⊢ Continuous (Function.HasUncurry.uncurry fun t => (f t).trans (g t))
  -/
  exact trans_continuous_family _ (continuous_uncurry_iff.mpr hf) _ (continuous_uncurry_iff.mpr hg)
  /-
    🎉 no goals
  -/


@[continuity]
theorem continuous_trans {x y z : X} : Continuous fun ρ : Path x y × Path y z => ρ.1.trans ρ.2 :=
  continuous_fst.path_trans continuous_snd


/-- Given a path in `X` and a path in `Y`, we can take their pointwise product to get a path in
`X × Y`. -/
protected def prod (γ₁ : Path a₁ a₂) (γ₂ : Path b₁ b₂) : Path (a₁, b₁) (a₂, b₂) where
  toContinuousMap := ContinuousMap.prodMk γ₁.toContinuousMap γ₂.toContinuousMap
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ : Path x y
                  a₁ a₂ a₃ : X
                  b₁ b₂ b₃ : Y
                  γ₁ : Path a₁ a₂
                  γ₂ : Path b₁ b₂
                  ⊢ Eq ((γ₁.prodMk γ₂.toContinuousMap).toFun 0) { fst := a₁, snd := b₁ }
                -/
  source' := by simp
                /-
                  🎉 no goals
                -/
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ : Path x y
                  a₁ a₂ a₃ : X
                  b₁ b₂ b₃ : Y
                  γ₁ : Path a₁ a₂
                  γ₂ : Path b₁ b₂
                  ⊢ Eq ((γ₁.prodMk γ₂.toContinuousMap).toFun 1) { fst := a₂, snd := b₂ }
                -/
  target' := by simp
                /-
                  🎉 no goals
                -/


@[simp]
theorem prod_coe (γ₁ : Path a₁ a₂) (γ₂ : Path b₁ b₂) :
    ⇑(γ₁.prod γ₂) = fun t => (γ₁ t, γ₂ t) :=
  rfl


/-- Path composition commutes with products -/
theorem trans_prod_eq_prod_trans (γ₁ : Path a₁ a₂) (δ₁ : Path a₂ a₃) (γ₂ : Path b₁ b₂)
    (δ₂ : Path b₂ b₃) : (γ₁.prod γ₂).trans (δ₁.prod δ₂) = (γ₁.trans δ₁).prod (γ₂.trans δ₂) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    a₁ a₂ a₃ : X
    b₁ b₂ b₃ : Y
    γ₁ : Path a₁ a₂
    δ₁ : Path a₂ a₃
    γ₂ : Path b₁ b₂
    δ₂ : Path b₂ b₃
    ⊢ Eq ((γ₁.prod γ₂).trans (δ₁.prod δ₂)) ((γ₁.trans δ₁).prod (γ₂.trans δ₂))
  -/
  ext t <;>
  /-
    case a.h.fst
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    a₁ a₂ a₃ : X
    b₁ b₂ b₃ : Y
    γ₁ : Path a₁ a₂
    δ₁ : Path a₂ a₃
    γ₂ : Path b₁ b₂
    δ₂ : Path b₂ b₃
    t : ↑unitInterval
    ⊢ Eq (((γ₁.prod γ₂).trans (δ₁.prod δ₂)) t).1 (((γ₁.trans δ₁).prod (γ₂.trans δ₂ …
  -/
  unfold Path.trans <;>
  /-
    case a.h.fst
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    a₁ a₂ a₃ : X
    b₁ b₂ b₃ : Y
    γ₁ : Path a₁ a₂
    δ₁ : Path a₂ a₃
    γ₂ : Path b₁ b₂
    δ₂ : Path b₂ b₃
    t : ↑unitInterval
    ⊢ Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) ((γ₁.prod γ₂).e …
  -/
  simp only [Path.coe_mk_mk, Path.prod_coe, Function.comp_apply] <;>
  /-
    case a.h.fst
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    a₁ a₂ a₃ : X
    b₁ b₂ b₃ : Y
    γ₁ : Path a₁ a₂
    δ₁ : Path a₂ a₃
    γ₂ : Path b₁ b₂
    δ₂ : Path b₂ b₃
    t : ↑unitInterval
    ⊢ Eq (ite (LE.le (↑t) (1 / 2)) ((γ₁.prod γ₂).extend (HMul.hMul 2 ↑t)) ((δ₁.pro …
  -/
  split_ifs <;>
  /-
    case pos
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    a₁ a₂ a₃ : X
    b₁ b₂ b₃ : Y
    γ₁ : Path a₁ a₂
    δ₁ : Path a₂ a₃
    γ₂ : Path b₁ b₂
    δ₂ : Path b₂ b₃
    t : ↑unitInterval
    h✝ : LE.le (↑t) (1 / 2)
    ⊢ Eq ((γ₁.prod γ₂).extend (HMul.hMul 2 ↑t)).1 (γ₁.extend (HMul.hMul 2 ↑t))
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a family of paths, one in each Xᵢ, we take their pointwise product to get a path in
Π i, Xᵢ. -/
protected def pi (γ : ∀ i, Path (as i) (bs i)) : Path as bs where
  toContinuousMap := ContinuousMap.pi fun i => (γ i).toContinuousMap
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝² : TopologicalSpace X
                  inst✝¹ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ : Path x y
                  χ : ι → Type u_4
                  inst✝ : (i : ι) → TopologicalSpace (χ i)
                  as bs cs : (i : ι) → χ i
                  γ : (i : ι) → Path (as i) (bs i)
                  ⊢ Eq ((ContinuousMap.pi fun i => (γ i).toContinuousMap).toFun 0) as
                -/
  source' := by simp
                /-
                  🎉 no goals
                -/
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝² : TopologicalSpace X
                  inst✝¹ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ : Path x y
                  χ : ι → Type u_4
                  inst✝ : (i : ι) → TopologicalSpace (χ i)
                  as bs cs : (i : ι) → χ i
                  γ : (i : ι) → Path (as i) (bs i)
                  ⊢ Eq ((ContinuousMap.pi fun i => (γ i).toContinuousMap).toFun 1) bs
                -/
  target' := by simp
                /-
                  🎉 no goals
                -/


@[simp]
theorem pi_coe (γ : ∀ i, Path (as i) (bs i)) : ⇑(Path.pi γ) = fun t i => γ i t :=
  rfl


/-- Path composition commutes with products -/
theorem trans_pi_eq_pi_trans (γ₀ : ∀ i, Path (as i) (bs i)) (γ₁ : ∀ i, Path (bs i) (cs i)) :
    (Path.pi γ₀).trans (Path.pi γ₁) = Path.pi fun i => (γ₀ i).trans (γ₁ i) := by
  /-
    ι : Type u_3
    χ : ι → Type u_4
    inst✝ : (i : ι) → TopologicalSpace (χ i)
    as bs cs : (i : ι) → χ i
    γ₀ : (i : ι) → Path (as i) (bs i)
    γ₁ : (i : ι) → Path (bs i) (cs i)
    ⊢ Eq ((Path.pi γ₀).trans (Path.pi γ₁)) (Path.pi fun i => (γ₀ i).trans (γ₁ i))
  -/
  ext t i
  /-
    case a.h.h
    ι : Type u_3
    χ : ι → Type u_4
    inst✝ : (i : ι) → TopologicalSpace (χ i)
    as bs cs : (i : ι) → χ i
    γ₀ : (i : ι) → Path (as i) (bs i)
    γ₁ : (i : ι) → Path (bs i) (cs i)
    t : ↑unitInterval
    i : ι
    ⊢ Eq (((Path.pi γ₀).trans (Path.pi γ₁)) t i) ((Path.pi fun i => (γ₀ i).trans ( …
  -/
  unfold Path.trans
  /-
    case a.h.h
    ι : Type u_3
    χ : ι → Type u_4
    inst✝ : (i : ι) → TopologicalSpace (χ i)
    as bs cs : (i : ι) → χ i
    γ₀ : (i : ι) → Path (as i) (bs i)
    γ₁ : (i : ι) → Path (bs i) (cs i)
    t : ↑unitInterval
    i : ι
    ⊢ Eq ({ toFun := Function.comp (fun t => ite (LE.le t (1 / 2)) ((Path.pi γ₀).e …
  -/
  simp only [Path.coe_mk_mk, Function.comp_apply, pi_coe]
  /-
    case a.h.h
    ι : Type u_3
    χ : ι → Type u_4
    inst✝ : (i : ι) → TopologicalSpace (χ i)
    as bs cs : (i : ι) → χ i
    γ₀ : (i : ι) → Path (as i) (bs i)
    γ₁ : (i : ι) → Path (bs i) (cs i)
    t : ↑unitInterval
    i : ι
    ⊢ Eq (ite (LE.le (↑t) (1 / 2)) ((Path.pi γ₀).extend (HMul.hMul 2 ↑t)) ((Path.p …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> rfl
                /-
                  🎉 no goals
                -/


/-- Pointwise multiplication of paths in a topological group. The additive version is probably more
useful. -/
@[to_additive "Pointwise addition of paths in a topological additive group."]
protected def mul [Mul X] [ContinuousMul X] {a₁ b₁ a₂ b₂ : X} (γ₁ : Path a₁ b₁) (γ₂ : Path a₂ b₂) :
    Path (a₁ * a₂) (b₁ * b₂) :=
  (γ₁.prod γ₂).map continuous_mul


@[to_additive]
protected theorem mul_apply [Mul X] [ContinuousMul X] {a₁ b₁ a₂ b₂ : X} (γ₁ : Path a₁ b₁)
    (γ₂ : Path a₂ b₂) (t : unitInterval) : (γ₁.mul γ₂) t = γ₁ t * γ₂ t :=
  rfl


/-- `γ.truncate t₀ t₁` is the path which follows the path `γ` on the
  time interval `[t₀, t₁]` and stays still otherwise. -/
def truncate {X : Type*} [TopologicalSpace X] {a b : X} (γ : Path a b) (t₀ t₁ : ℝ) :
    Path (γ.extend <| min t₀ t₁) (γ.extend t₁) where
  toFun s := γ.extend (min (max s t₀) t₁)
  continuous_toFun :=
    γ.continuous_extend.comp ((continuous_subtype_val.max continuous_const).min continuous_const)
  source' := by
    /-
      X✝ : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X✝
      inst✝¹ : TopologicalSpace Y
      x y z : X✝
      ι : Type u_3
      γ✝ : Path x y
      X : Type u_4
      inst✝ : TopologicalSpace X
      a b : X
      γ : Path a b
      t₀ t₁ : Real
      ⊢ Eq ({ toFun := fun s => γ.extend (Min.min (Max.max (↑s) t₀) t₁), continuous_ …
    -/
    simp only [min_def, max_def']
    /-
      X✝ : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X✝
      inst✝¹ : TopologicalSpace Y
      x y z : X✝
      ι : Type u_3
      γ✝ : Path x y
      X : Type u_4
      inst✝ : TopologicalSpace X
      a b : X
      γ : Path a b
      t₀ t₁ : Real
      ⊢ Eq (γ.extend (ite (LE.le (ite (LE.le t₀ ↑0) (↑0) t₀) t₁) (ite (LE.le t₀ ↑0)  …
    -/
    split_ifs with h₁ h₂ h₃ h₄
      /-
        case pos
        X✝ : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X✝
        inst✝¹ : TopologicalSpace Y
        x y z : X✝
        ι : Type u_3
        γ✝ : Path x y
        X : Type u_4
        inst✝ : TopologicalSpace X
        a b : X
        γ : Path a b
        t₀ t₁ : Real
        h₁ : LE.le t₀ ↑0
        h₂ : LE.le (↑0) t₁
        h₃ : LE.le t₀ t₁
        ⊢ Eq (γ.extend ↑0) (γ.extend t₀)
      -/
    · simp [γ.extend_of_le_zero h₁]
      /-
        🎉 no goals
      -/
      /-
        case neg
        X✝ : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X✝
        inst✝¹ : TopologicalSpace Y
        x y z : X✝
        ι : Type u_3
        γ✝ : Path x y
        X : Type u_4
        inst✝ : TopologicalSpace X
        a b : X
        γ : Path a b
        t₀ t₁ : Real
        h₁ : LE.le t₀ ↑0
        h₂ : LE.le (↑0) t₁
        h₃ : Not (LE.le t₀ t₁)
        ⊢ Eq (γ.extend ↑0) (γ.extend t₁)
      -/
    · congr
      /-
        case neg.e_a
        X✝ : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X✝
        inst✝¹ : TopologicalSpace Y
        x y z : X✝
        ι : Type u_3
        γ✝ : Path x y
        X : Type u_4
        inst✝ : TopologicalSpace X
        a b : X
        γ : Path a b
        t₀ t₁ : Real
        h₁ : LE.le t₀ ↑0
        h₂ : LE.le (↑0) t₁
        h₃ : Not (LE.le t₀ t₁)
        ⊢ Eq (↑0) t₁
      -/
      linarith
      /-
        🎉 no goals
      -/
      /-
        case pos
        X✝ : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X✝
        inst✝¹ : TopologicalSpace Y
        x y z : X✝
        ι : Type u_3
        γ✝ : Path x y
        X : Type u_4
        inst✝ : TopologicalSpace X
        a b : X
        γ : Path a b
        t₀ t₁ : Real
        h₁ : LE.le t₀ ↑0
        h₂ : Not (LE.le (↑0) t₁)
        h₄ : LE.le t₀ t₁
        ⊢ Eq (γ.extend t₁) (γ.extend t₀)
      -/
    · have h₄ : t₁ ≤ 0 := le_of_lt (by simpa using h₂)
      /-
        case pos
        X✝ : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X✝
        inst✝¹ : TopologicalSpace Y
        x y z : X✝
        ι : Type u_3
        γ✝ : Path x y
        X : Type u_4
        inst✝ : TopologicalSpace X
        a b : X
        γ : Path a b
        t₀ t₁ : Real
        h₁ : LE.le t₀ ↑0
        h₂ : Not (LE.le (↑0) t₁)
        h₄✝ : LE.le t₀ t₁
        h₄ : LE.le t₁ 0
        ⊢ Eq (γ.extend t₁) (γ.extend t₀)
      -/
      simp [γ.extend_of_le_zero h₄, γ.extend_of_le_zero h₁]
      /-
        🎉 no goals
      -/
    /-
      case neg
      X✝ : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X✝
      inst✝¹ : TopologicalSpace Y
      x y z : X✝
      ι : Type u_3
      γ✝ : Path x y
      X : Type u_4
      inst✝ : TopologicalSpace X
      a b : X
      γ : Path a b
      t₀ t₁ : Real
      h₁ : LE.le t₀ ↑0
      h₂ : Not (LE.le (↑0) t₁)
      h₄ : Not (LE.le t₀ t₁)
      ⊢ Eq (γ.extend t₁) (γ.extend t₁)
    -/
    all_goals rfl
    /-
      🎉 no goals
    -/
  target' := by
    /-
      X✝ : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X✝
      inst✝¹ : TopologicalSpace Y
      x y z : X✝
      ι : Type u_3
      γ✝ : Path x y
      X : Type u_4
      inst✝ : TopologicalSpace X
      a b : X
      γ : Path a b
      t₀ t₁ : Real
      ⊢ Eq ({ toFun := fun s => γ.extend (Min.min (Max.max (↑s) t₀) t₁), continuous_ …
    -/
    simp only [min_def, max_def']
    /-
      X✝ : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X✝
      inst✝¹ : TopologicalSpace Y
      x y z : X✝
      ι : Type u_3
      γ✝ : Path x y
      X : Type u_4
      inst✝ : TopologicalSpace X
      a b : X
      γ : Path a b
      t₀ t₁ : Real
      ⊢ Eq (γ.extend (ite (LE.le (ite (LE.le t₀ ↑1) (↑1) t₀) t₁) (ite (LE.le t₀ ↑1)  …
    -/
    split_ifs with h₁ h₂ h₃
      /-
        case pos
        X✝ : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X✝
        inst✝¹ : TopologicalSpace Y
        x y z : X✝
        ι : Type u_3
        γ✝ : Path x y
        X : Type u_4
        inst✝ : TopologicalSpace X
        a b : X
        γ : Path a b
        t₀ t₁ : Real
        h₁ : LE.le t₀ ↑1
        h₂ : LE.le (↑1) t₁
        ⊢ Eq (γ.extend ↑1) (γ.extend t₁)
      -/
    · simp [γ.extend_of_one_le h₂]
      /-
        🎉 no goals
      -/
      /-
        case neg
        X✝ : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X✝
        inst✝¹ : TopologicalSpace Y
        x y z : X✝
        ι : Type u_3
        γ✝ : Path x y
        X : Type u_4
        inst✝ : TopologicalSpace X
        a b : X
        γ : Path a b
        t₀ t₁ : Real
        h₁ : LE.le t₀ ↑1
        h₂ : Not (LE.le (↑1) t₁)
        ⊢ Eq (γ.extend t₁) (γ.extend t₁)
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case pos
        X✝ : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X✝
        inst✝¹ : TopologicalSpace Y
        x y z : X✝
        ι : Type u_3
        γ✝ : Path x y
        X : Type u_4
        inst✝ : TopologicalSpace X
        a b : X
        γ : Path a b
        t₀ t₁ : Real
        h₁ : Not (LE.le t₀ ↑1)
        h₃ : LE.le t₀ t₁
        ⊢ Eq (γ.extend t₀) (γ.extend t₁)
      -/
    · have h₄ : 1 ≤ t₀ := le_of_lt (by simpa using h₁)
      /-
        case pos
        X✝ : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X✝
        inst✝¹ : TopologicalSpace Y
        x y z : X✝
        ι : Type u_3
        γ✝ : Path x y
        X : Type u_4
        inst✝ : TopologicalSpace X
        a b : X
        γ : Path a b
        t₀ t₁ : Real
        h₁ : Not (LE.le t₀ ↑1)
        h₃ : LE.le t₀ t₁
        h₄ : LE.le 1 t₀
        ⊢ Eq (γ.extend t₀) (γ.extend t₁)
      -/
      simp [γ.extend_of_one_le h₄, γ.extend_of_one_le (h₄.trans h₃)]
      /-
        🎉 no goals
      -/
      /-
        case neg
        X✝ : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X✝
        inst✝¹ : TopologicalSpace Y
        x y z : X✝
        ι : Type u_3
        γ✝ : Path x y
        X : Type u_4
        inst✝ : TopologicalSpace X
        a b : X
        γ : Path a b
        t₀ t₁ : Real
        h₁ : Not (LE.le t₀ ↑1)
        h₃ : Not (LE.le t₀ t₁)
        ⊢ Eq (γ.extend t₁) (γ.extend t₁)
      -/
    · rfl
      /-
        🎉 no goals
      -/


/-- `γ.truncateOfLE t₀ t₁ h`, where `h : t₀ ≤ t₁` is `γ.truncate t₀ t₁`
  casted as a path from `γ.extend t₀` to `γ.extend t₁`. -/
def truncateOfLE {X : Type*} [TopologicalSpace X] {a b : X} (γ : Path a b) {t₀ t₁ : ℝ}
    (h : t₀ ≤ t₁) : Path (γ.extend t₀) (γ.extend t₁) :=
                              /-
                                X✝ : Type u_1
                                Y : Type u_2
                                inst✝² : TopologicalSpace X✝
                                inst✝¹ : TopologicalSpace Y
                                x y z : X✝
                                ι : Type u_3
                                γ✝ : Path x y
                                X : Type u_4
                                inst✝ : TopologicalSpace X
                                a b : X
                                γ : Path a b
                                t₀ t₁ : Real
                                h : LE.le t₀ t₁
                                ⊢ Eq (γ.extend t₀) (γ.extend (Min.min t₀ t₁))
                              -/
  (γ.truncate t₀ t₁).cast (by rw [min_eq_left h]) rfl
                              /-
                                🎉 no goals
                              -/


theorem truncate_range {a b : X} (γ : Path a b) {t₀ t₁ : ℝ} :
    range (γ.truncate t₀ t₁) ⊆ range γ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    t₀ t₁ : Real
    ⊢ HasSubset.Subset (Set.range ⇑(γ.truncate t₀ t₁)) (Set.range ⇑γ)
  -/
  rw [← γ.extend_range]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    t₀ t₁ : Real
    ⊢ HasSubset.Subset (Set.range ⇑(γ.truncate t₀ t₁)) (Set.range γ.extend)
  -/
  simp only [range_subset_iff, SetCoe.exists, SetCoe.forall]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    t₀ t₁ : Real
    ⊢ ∀ (x : Real) (h : Membership.mem unitInterval x), Membership.mem (Set.range  …
  -/
  intro x _hx
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    t₀ t₁ x : Real
    _hx : Membership.mem unitInterval x
    ⊢ Membership.mem (Set.range γ.extend) ((γ.truncate t₀ t₁) ⟨x, _hx⟩)
  -/
  simp only [DFunLike.coe, Path.truncate, mem_range_self]
  /-
    🎉 no goals
  -/


/-- For a path `γ`, `γ.truncate` gives a "continuous family of paths", by which we
  mean the uncurried function which maps `(t₀, t₁, s)` to `γ.truncate t₀ t₁ s` is continuous. -/
@[continuity]
theorem truncate_continuous_family {a b : X} (γ : Path a b) :
    Continuous (fun x => γ.truncate x.1 x.2.1 x.2.2 : ℝ × ℝ × I → X) :=
  γ.continuous_extend.comp
    (((continuous_subtype_val.comp (continuous_snd.comp continuous_snd)).max continuous_fst).min
      (continuous_fst.comp continuous_snd))


@[continuity]
theorem truncate_const_continuous_family {a b : X} (γ : Path a b)
    (t : ℝ) : Continuous ↿(γ.truncate t) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    t : Real
    ⊢ Continuous (Function.HasUncurry.uncurry (γ.truncate t))
  -/
  have key : Continuous (fun x => (t, x) : ℝ × I → ℝ × ℝ × I) := by fun_prop
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    t : Real
    key : Continuous fun x => { fst := t, snd := x }
    ⊢ Continuous (Function.HasUncurry.uncurry (γ.truncate t))
  -/
  exact γ.truncate_continuous_family.comp key
  /-
    🎉 no goals
  -/


@[simp]
theorem truncate_self {a b : X} (γ : Path a b) (t : ℝ) :
                                                        /-
                                                          X : Type u_1
                                                          Y : Type u_2
                                                          inst✝¹ : TopologicalSpace X
                                                          inst✝ : TopologicalSpace Y
                                                          x y z : X
                                                          ι : Type u_3
                                                          γ✝ : Path x y
                                                          a b : X
                                                          γ : Path a b
                                                          t : Real
                                                          ⊢ Eq (γ.extend (Min.min t t)) (γ.extend t)
                                                        -/
    γ.truncate t t = (Path.refl <| γ.extend t).cast (by rw [min_self]) rfl := by
                                                        /-
                                                          🎉 no goals
                                                        -/
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    t : Real
    ⊢ Eq (γ.truncate t t) ((Path.refl (γ.extend t)).cast ⋯ ⋯)
  -/
  ext x
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    t : Real
    x : ↑unitInterval
    ⊢ Eq ((γ.truncate t t) x) (((Path.refl (γ.extend t)).cast ⋯ ⋯) x)
  -/
  rw [cast_coe]
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    t : Real
    x : ↑unitInterval
    ⊢ Eq ((γ.truncate t t) x) ((Path.refl (γ.extend t)) x)
  -/
  simp only [truncate, DFunLike.coe, refl, min_def, max_def]
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    t : Real
    x : ↑unitInterval
    ⊢ Eq (γ.extend (ite (LE.le (ite (LE.le (↑x) t) t ↑x) t) (ite (LE.le (↑x) t) t  …
  -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
  split_ifs with h₁ h₂ <;> congr
                           /-
                             🎉 no goals
                           -/


@[simp 1001] -- Porting note: increase `simp` priority so left-hand side doesn't simplify
theorem truncate_zero_zero {a b : X} (γ : Path a b) :
                                            /-
                                              X : Type u_1
                                              Y : Type u_2
                                              inst✝¹ : TopologicalSpace X
                                              inst✝ : TopologicalSpace Y
                                              x y z : X
                                              ι : Type u_3
                                              γ✝ : Path x y
                                              a b : X
                                              γ : Path a b
                                              ⊢ Eq (γ.extend (Min.min 0 0)) a
                                            -/
    γ.truncate 0 0 = (Path.refl a).cast (by rw [min_self, γ.extend_zero]) γ.extend_zero := by
                                            /-
                                              🎉 no goals
                                            -/
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    ⊢ Eq (γ.truncate 0 0) ((Path.refl a).cast ⋯ ⋯)
  -/
  convert γ.truncate_self 0
  /-
    🎉 no goals
  -/


@[simp 1001] -- Porting note: increase `simp` priority so left-hand side doesn't simplify
theorem truncate_one_one {a b : X} (γ : Path a b) :
                                            /-
                                              X : Type u_1
                                              Y : Type u_2
                                              inst✝¹ : TopologicalSpace X
                                              inst✝ : TopologicalSpace Y
                                              x y z : X
                                              ι : Type u_3
                                              γ✝ : Path x y
                                              a b : X
                                              γ : Path a b
                                              ⊢ Eq (γ.extend (Min.min 1 1)) b
                                            -/
    γ.truncate 1 1 = (Path.refl b).cast (by rw [min_self, γ.extend_one]) γ.extend_one := by
                                            /-
                                              🎉 no goals
                                            -/
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    ⊢ Eq (γ.truncate 1 1) ((Path.refl b).cast ⋯ ⋯)
  -/
  convert γ.truncate_self 1
  /-
    🎉 no goals
  -/


@[simp]
theorem truncate_zero_one {a b : X} (γ : Path a b) :
                                /-
                                  X : Type u_1
                                  Y : Type u_2
                                  inst✝¹ : TopologicalSpace X
                                  inst✝ : TopologicalSpace Y
                                  x y z : X
                                  ι : Type u_3
                                  γ✝ : Path x y
                                  a b : X
                                  γ : Path a b
                                  ⊢ Eq (γ.extend (Min.min 0 1)) a
                                -/
                                /-
                                  🎉 no goals
                                -/
    γ.truncate 0 1 = γ.cast (by simp [zero_le_one, extend_zero]) (by simp) := by
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    ⊢ Eq (γ.truncate 0 1) (γ.cast ⋯ ⋯)
  -/
  ext x
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    x : ↑unitInterval
    ⊢ Eq ((γ.truncate 0 1) x) ((γ.cast ⋯ ⋯) x)
  -/
  rw [cast_coe]
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    x : ↑unitInterval
    ⊢ Eq ((γ.truncate 0 1) x) (γ x)
  -/
  have : ↑x ∈ (Icc 0 1 : Set ℝ) := x.2
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    a b : X
    γ : Path a b
    x : ↑unitInterval
    this : Membership.mem (Set.Icc 0 1) ↑x
    ⊢ Eq ((γ.truncate 0 1) x) (γ x)
  -/
  rw [truncate, coe_mk_mk, max_eq_left this.1, min_eq_left this.2, extend_extends']
  /-
    🎉 no goals
  -/


/-- Given a path `γ` and a function `f : I → I` where `f 0 = 0` and `f 1 = 1`, `γ.reparam f` is the
path defined by `γ ∘ f`.
-/
def reparam (γ : Path x y) (f : I → I) (hfcont : Continuous f) (hf₀ : f 0 = 0) (hf₁ : f 1 = 1) :
    Path x y where
  toFun := γ ∘ f
                         /-
                           X : Type u_1
                           Y : Type u_2
                           inst✝¹ : TopologicalSpace X
                           inst✝ : TopologicalSpace Y
                           x y z : X
                           ι : Type u_3
                           γ✝ γ : Path x y
                           f : ↑unitInterval → ↑unitInterval
                           hfcont : Continuous f
                           hf₀ : Eq (f 0) 0
                           hf₁ : Eq (f 1) 1
                           ⊢ Continuous (Function.comp (⇑γ) f)
                         -/
  continuous_toFun := by fun_prop
                         /-
                           🎉 no goals
                         -/
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ γ : Path x y
                  f : ↑unitInterval → ↑unitInterval
                  hfcont : Continuous f
                  hf₀ : Eq (f 0) 0
                  hf₁ : Eq (f 1) 1
                  ⊢ Eq ({ toFun := Function.comp (⇑γ) f, continuous_toFun := ⋯ }.toFun 0) x
                -/
  source' := by simp [hf₀]
                /-
                  🎉 no goals
                -/
                /-
                  X : Type u_1
                  Y : Type u_2
                  inst✝¹ : TopologicalSpace X
                  inst✝ : TopologicalSpace Y
                  x y z : X
                  ι : Type u_3
                  γ✝ γ : Path x y
                  f : ↑unitInterval → ↑unitInterval
                  hfcont : Continuous f
                  hf₀ : Eq (f 0) 0
                  hf₁ : Eq (f 1) 1
                  ⊢ Eq ({ toFun := Function.comp (⇑γ) f, continuous_toFun := ⋯ }.toFun 1) y
                -/
  target' := by simp [hf₁]
                /-
                  🎉 no goals
                -/


@[simp]
theorem coe_reparam (γ : Path x y) {f : I → I} (hfcont : Continuous f) (hf₀ : f 0 = 0)
    (hf₁ : f 1 = 1) : ⇑(γ.reparam f hfcont hf₀ hf₁) = γ ∘ f :=
  rfl
-- Porting note: this seems like it was poorly named (was: `coe_to_fun`)


@[simp]
theorem reparam_id (γ : Path x y) : γ.reparam id continuous_id rfl rfl = γ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    ⊢ Eq (γ.reparam id ⋯ ⋯ ⋯) γ
  -/
  ext
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    x✝ : ↑unitInterval
    ⊢ Eq ((γ.reparam id ⋯ ⋯ ⋯) x✝) (γ x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem range_reparam (γ : Path x y) {f : I → I} (hfcont : Continuous f) (hf₀ : f 0 = 0)
    (hf₁ : f 1 = 1) : range (γ.reparam f hfcont hf₀ hf₁) = range γ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    f : ↑unitInterval → ↑unitInterval
    hfcont : Continuous f
    hf₀ : Eq (f 0) 0
    hf₁ : Eq (f 1) 1
    ⊢ Eq (Set.range ⇑(γ.reparam f hfcont hf₀ hf₁)) (Set.range ⇑γ)
  -/
  change range (γ ∘ f) = range γ
  have : range f = univ := by
    rw [range_eq_univ]
    intro t
    have h₁ : Continuous (Set.IccExtend (zero_le_one' ℝ) f) := by continuity
    have := intermediate_value_Icc (zero_le_one' ℝ) h₁.continuousOn
    · rw [IccExtend_left, IccExtend_right, Icc.mk_zero, Icc.mk_one, hf₀, hf₁] at this
      rcases this t.2 with ⟨w, hw₁, hw₂⟩
      rw [IccExtend_of_mem _ _ hw₁] at hw₂
      exact ⟨_, hw₂⟩
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    γ : Path x y
    f : ↑unitInterval → ↑unitInterval
    hfcont : Continuous f
    hf₀ : Eq (f 0) 0
    hf₁ : Eq (f 1) 1
    this : Eq (Set.range f) Set.univ
    ⊢ Eq (Set.range (Function.comp (⇑γ) f)) (Set.range ⇑γ)
  -/
  rw [range_comp, this, image_univ]
  /-
    🎉 no goals
  -/


theorem refl_reparam {f : I → I} (hfcont : Continuous f) (hf₀ : f 0 = 0) (hf₁ : f 1 = 1) :
    (refl x).reparam f hfcont hf₀ hf₁ = refl x := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x : X
    f : ↑unitInterval → ↑unitInterval
    hfcont : Continuous f
    hf₀ : Eq (f 0) 0
    hf₁ : Eq (f 1) 1
    ⊢ Eq ((Path.refl x).reparam f hfcont hf₀ hf₁) (Path.refl x)
  -/
  ext
  /-
    case a.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    x : X
    f : ↑unitInterval → ↑unitInterval
    hfcont : Continuous f
    hf₀ : Eq (f 0) 0
    hf₁ : Eq (f 1) 1
    x✝ : ↑unitInterval
    ⊢ Eq (((Path.refl x).reparam f hfcont hf₀ hf₁) x✝) ((Path.refl x) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The relation "being joined by a path". This is an equivalence relation. -/
def Joined (x y : X) : Prop :=
  Nonempty (Path x y)


@[refl]
theorem Joined.refl (x : X) : Joined x x :=
  ⟨Path.refl x⟩


/-- When two points are joined, choose some path from `x` to `y`. -/
def Joined.somePath (h : Joined x y) : Path x y :=
  Nonempty.some h


@[symm]
theorem Joined.symm {x y : X} (h : Joined x y) : Joined y x :=
  ⟨h.somePath.symm⟩


@[trans]
theorem Joined.trans {x y z : X} (hxy : Joined x y) (hyz : Joined y z) : Joined x z :=
  ⟨hxy.somePath.trans hyz.somePath⟩


/-- The setoid corresponding the equivalence relation of being joined by a continuous path. -/
def pathSetoid : Setoid X where
  r := Joined
  iseqv := Equivalence.mk Joined.refl Joined.symm Joined.trans


/-- The quotient type of points of a topological space modulo being joined by a continuous path. -/
def ZerothHomotopy :=
  Quotient (pathSetoid X)


instance ZerothHomotopy.inhabited : Inhabited (ZerothHomotopy ℝ) :=
  ⟨@Quotient.mk' ℝ (pathSetoid ℝ) 0⟩


/-- The relation "being joined by a path in `F`". Not quite an equivalence relation since it's not
reflexive for points that do not belong to `F`. -/
def JoinedIn (F : Set X) (x y : X) : Prop :=
  ∃ γ : Path x y, ∀ t, γ t ∈ F


theorem JoinedIn.mem (h : JoinedIn F x y) : x ∈ F ∧ y ∈ F := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    F : Set X
    h : JoinedIn F x y
    ⊢ And (Membership.mem F x) (Membership.mem F y)
  -/
  rcases h with ⟨γ, γ_in⟩
  /-
    case intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    F : Set X
    γ : Path x y
    γ_in : ∀ (t : ↑unitInterval), Membership.mem F (γ t)
    ⊢ And (Membership.mem F x) (Membership.mem F y)
  -/
  have : γ 0 ∈ F ∧ γ 1 ∈ F := by constructor <;> apply γ_in
  /-
    case intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    F : Set X
    γ : Path x y
    γ_in : ∀ (t : ↑unitInterval), Membership.mem F (γ t)
    this : And (Membership.mem F (γ 0)) (Membership.mem F (γ 1))
    ⊢ And (Membership.mem F x) (Membership.mem F y)
  -/
  simpa using this
  /-
    🎉 no goals
  -/


theorem JoinedIn.source_mem (h : JoinedIn F x y) : x ∈ F :=
  h.mem.1


theorem JoinedIn.target_mem (h : JoinedIn F x y) : y ∈ F :=
  h.mem.2


/-- When `x` and `y` are joined in `F`, choose a path from `x` to `y` inside `F` -/
def JoinedIn.somePath (h : JoinedIn F x y) : Path x y :=
  Classical.choose h


theorem JoinedIn.somePath_mem (h : JoinedIn F x y) (t : I) : h.somePath t ∈ F :=
  Classical.choose_spec h t


/-- If `x` and `y` are joined in the set `F`, then they are joined in the subtype `F`. -/
theorem JoinedIn.joined_subtype (h : JoinedIn F x y) :
    Joined (⟨x, h.source_mem⟩ : F) (⟨y, h.target_mem⟩ : F) :=
  ⟨{  toFun := fun t => ⟨h.somePath t, h.somePath_mem t⟩
                             /-
                               X : Type u_1
                               inst✝ : TopologicalSpace X
                               x y : X
                               F : Set X
                               h : JoinedIn F x y
                               ⊢ Continuous fun t => ⟨h.somePath t, ⋯⟩
                             -/
      continuous_toFun := by fun_prop
                             /-
                               🎉 no goals
                             -/
                    /-
                      X : Type u_1
                      inst✝ : TopologicalSpace X
                      x y : X
                      F : Set X
                      h : JoinedIn F x y
                      ⊢ Eq ({ toFun := fun t => ⟨h.somePath t, ⋯⟩, continuous_toFun := ⋯ }.toFun 0)  …
                    -/
      source' := by simp
                    /-
                      🎉 no goals
                    -/
                    /-
                      X : Type u_1
                      inst✝ : TopologicalSpace X
                      x y : X
                      F : Set X
                      h : JoinedIn F x y
                      ⊢ Eq ({ toFun := fun t => ⟨h.somePath t, ⋯⟩, continuous_toFun := ⋯ }.toFun 1)  …
                    -/
      target' := by simp }⟩
                    /-
                      🎉 no goals
                    -/


theorem JoinedIn.ofLine {f : ℝ → X} (hf : ContinuousOn f I) (h₀ : f 0 = x) (h₁ : f 1 = y)
    (hF : f '' I ⊆ F) : JoinedIn F x y :=
  ⟨Path.ofLine hf h₀ h₁, fun t => hF <| Path.ofLine_mem hf h₀ h₁ t⟩


theorem JoinedIn.joined (h : JoinedIn F x y) : Joined x y :=
  ⟨h.somePath⟩


theorem joinedIn_iff_joined (x_in : x ∈ F) (y_in : y ∈ F) :
    JoinedIn F x y ↔ Joined (⟨x, x_in⟩ : F) (⟨y, y_in⟩ : F) :=
                                                                                  /-
                                                                                    X : Type u_1
                                                                                    inst✝ : TopologicalSpace X
                                                                                    x y : X
                                                                                    F : Set X
                                                                                    x_in : Membership.mem F x
                                                                                    y_in : Membership.mem F y
                                                                                    h : Joined ⟨x, x_in⟩ ⟨y, y_in⟩
                                                                                    ⊢ ∀ (t : ↑unitInterval), Membership.mem F ((h.somePath.map ⋯) t)
                                                                                  -/
  ⟨fun h => h.joined_subtype, fun h => ⟨h.somePath.map continuous_subtype_val, by simp⟩⟩
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
theorem joinedIn_univ : JoinedIn univ x y ↔ Joined x y := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    ⊢ Iff (JoinedIn Set.univ x y) (Joined x y)
  -/
  simp [JoinedIn, Joined, exists_true_iff_nonempty]
  /-
    🎉 no goals
  -/


theorem JoinedIn.mono {U V : Set X} (h : JoinedIn U x y) (hUV : U ⊆ V) : JoinedIn V x y :=
  ⟨h.somePath, fun t => hUV (h.somePath_mem t)⟩


theorem JoinedIn.refl (h : x ∈ F) : JoinedIn F x x :=
  ⟨Path.refl x, fun _t => h⟩


@[symm]
theorem JoinedIn.symm (h : JoinedIn F x y) : JoinedIn F y x := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    F : Set X
    h : JoinedIn F x y
    ⊢ JoinedIn F y x
  -/
  cases' h.mem with hx hy
  /-
    case intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    F : Set X
    h : JoinedIn F x y
    hx : Membership.mem F x
    hy : Membership.mem F y
    ⊢ JoinedIn F y x
  -/
  simp_all only [joinedIn_iff_joined]
  /-
    case intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    F : Set X
    hx✝ : Membership.mem F x
    hy✝ : Membership.mem F y
    h : Joined ⟨x, ⋯⟩ ⟨y, ⋯⟩
    hx : Membership.mem F x
    hy : Membership.mem F y
    ⊢ Joined ⟨y, ⋯⟩ ⟨x, ⋯⟩
  -/
  exact h.symm
  /-
    🎉 no goals
  -/


theorem JoinedIn.trans (hxy : JoinedIn F x y) (hyz : JoinedIn F y z) : JoinedIn F x z := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y z : X
    F : Set X
    hxy : JoinedIn F x y
    hyz : JoinedIn F y z
    ⊢ JoinedIn F x z
  -/
  cases' hxy.mem with hx hy
  /-
    case intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y z : X
    F : Set X
    hxy : JoinedIn F x y
    hyz : JoinedIn F y z
    hx : Membership.mem F x
    hy : Membership.mem F y
    ⊢ JoinedIn F x z
  -/
  cases' hyz.mem with hx hy
  /-
    case intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y z : X
    F : Set X
    hxy : JoinedIn F x y
    hyz : JoinedIn F y z
    hx✝ : Membership.mem F x
    hy✝ hx : Membership.mem F y
    hy : Membership.mem F z
    ⊢ JoinedIn F x z
  -/
  simp_all only [joinedIn_iff_joined]
  /-
    case intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y z : X
    F : Set X
    hx✝¹ : Membership.mem F x
    hy✝¹ : Membership.mem F y
    hy✝ : Membership.mem F z
    hxy : Joined ⟨x, ⋯⟩ ⟨y, ⋯⟩
    hyz : Joined ⟨y, ⋯⟩ ⟨z, ⋯⟩
    hx✝ : Membership.mem F x
    hx : Membership.mem F y
    hy : Membership.mem F z
    ⊢ Joined ⟨x, ⋯⟩ ⟨z, ⋯⟩
  -/
  exact hxy.trans hyz
  /-
    🎉 no goals
  -/


theorem Specializes.joinedIn (h : x ⤳ y) (hx : x ∈ F) (hy : y ∈ F) : JoinedIn F x y := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    F : Set X
    h : Specializes x y
    hx : Membership.mem F x
    hy : Membership.mem F y
    ⊢ JoinedIn F x y
  -/
  refine ⟨⟨⟨Set.piecewise {1} (const I y) (const I x), ?_⟩, by simp, by simp⟩, fun t ↦ ?_⟩
  · exact isClosed_singleton.continuous_piecewise_of_specializes continuous_const continuous_const
      fun _ ↦ h
    /-
      case refine_2
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y : X
      F : Set X
      h : Specializes x y
      hx : Membership.mem F x
      hy : Membership.mem F y
      t : ↑unitInterval
      ⊢ Membership.mem F ({ toFun := (Singleton.singleton 1).piecewise (Function.con …
    -/
  · simp only [Path.coe_mk_mk, piecewise]
    /-
      case refine_2
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y : X
      F : Set X
      h : Specializes x y
      hx : Membership.mem F x
      hy : Membership.mem F y
      t : ↑unitInterval
      ⊢ Membership.mem F (ite (Membership.mem (Singleton.singleton 1) t) (Function.c …
    -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> assumption
                  /-
                    🎉 no goals
                  -/


theorem Inseparable.joinedIn (h : Inseparable x y) (hx : x ∈ F) (hy : y ∈ F) : JoinedIn F x y :=
  h.specializes.joinedIn hx hy


theorem JoinedIn.map_continuousOn (h : JoinedIn F x y) {f : X → Y} (hf : ContinuousOn f F) :
    JoinedIn (f '' F) (f x) (f y) :=
  let ⟨γ, hγ⟩ := h
  ⟨γ.map' <| hf.mono (range_subset_iff.mpr hγ), fun t ↦ mem_image_of_mem _ (hγ t)⟩


theorem JoinedIn.map (h : JoinedIn F x y) {f : X → Y} (hf : Continuous f) :
    JoinedIn (f '' F) (f x) (f y) :=
  h.map_continuousOn hf.continuousOn


theorem Topology.IsInducing.joinedIn_image {f : X → Y} (hf : IsInducing f) (hx : x ∈ F)
    (hy : y ∈ F) : JoinedIn (f '' F) (f x) (f y) ↔ JoinedIn F x y := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y : X
    F : Set X
    f : X → Y
    hf : Topology.IsInducing f
    hx : Membership.mem F x
    hy : Membership.mem F y
    ⊢ Iff (JoinedIn (Set.image f F) (f x) (f y)) (JoinedIn F x y)
  -/
  refine ⟨?_, (.map · hf.continuous)⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y : X
    F : Set X
    f : X → Y
    hf : Topology.IsInducing f
    hx : Membership.mem F x
    hy : Membership.mem F y
    ⊢ JoinedIn (Set.image f F) (f x) (f y) → JoinedIn F x y
  -/
  rintro ⟨γ, hγ⟩
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y : X
    F : Set X
    f : X → Y
    hf : Topology.IsInducing f
    hx : Membership.mem F x
    hy : Membership.mem F y
    γ : Path (f x) (f y)
    hγ : ∀ (t : ↑unitInterval), Membership.mem (Set.image f F) (γ t)
    ⊢ JoinedIn F x y
  -/
  choose γ' hγ'F hγ' using hγ
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y : X
    F : Set X
    f : X → Y
    hf : Topology.IsInducing f
    hx : Membership.mem F x
    hy : Membership.mem F y
    γ : Path (f x) (f y)
    γ' : ↑unitInterval → X
    hγ'F : ∀ (t : ↑unitInterval), Membership.mem F (γ' t)
    hγ' : ∀ (t : ↑unitInterval), Eq (f (γ' t)) (γ t)
    ⊢ JoinedIn F x y
  -/
  have h₀ : x ⤳ γ' 0 := by rw [← hf.specializes_iff, hγ', γ.source]
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y : X
    F : Set X
    f : X → Y
    hf : Topology.IsInducing f
    hx : Membership.mem F x
    hy : Membership.mem F y
    γ : Path (f x) (f y)
    γ' : ↑unitInterval → X
    hγ'F : ∀ (t : ↑unitInterval), Membership.mem F (γ' t)
    hγ' : ∀ (t : ↑unitInterval), Eq (f (γ' t)) (γ t)
    h₀ : Specializes x (γ' 0)
    ⊢ JoinedIn F x y
  -/
  have h₁ : γ' 1 ⤳ y := by rw [← hf.specializes_iff, hγ', γ.target]
  have h : JoinedIn F (γ' 0) (γ' 1) := by
    refine ⟨⟨⟨γ', ?_⟩, rfl, rfl⟩, hγ'F⟩
    simpa only [hf.continuous_iff, comp_def, hγ'] using map_continuous γ
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x y : X
    F : Set X
    f : X → Y
    hf : Topology.IsInducing f
    hx : Membership.mem F x
    hy : Membership.mem F y
    γ : Path (f x) (f y)
    γ' : ↑unitInterval → X
    hγ'F : ∀ (t : ↑unitInterval), Membership.mem F (γ' t)
    hγ' : ∀ (t : ↑unitInterval), Eq (f (γ' t)) (γ t)
    h₀ : Specializes x (γ' 0)
    h₁ : Specializes (γ' 1) y
    h : JoinedIn F (γ' 0) (γ' 1)
    ⊢ JoinedIn F x y
  -/
  exact (h₀.joinedIn hx (hγ'F _)).trans <| h.trans <| h₁.joinedIn (hγ'F _) hy
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")] alias Inducing.joinedIn_image := IsInducing.joinedIn_image


/-- The path component of `x` is the set of points that can be joined to `x`. -/
def pathComponent (x : X) :=
  { y | Joined x y }


theorem mem_pathComponent_iff : x ∈ pathComponent y ↔ Joined y x := .rfl


@[simp]
theorem mem_pathComponent_self (x : X) : x ∈ pathComponent x :=
  Joined.refl x


@[simp]
theorem pathComponent.nonempty (x : X) : (pathComponent x).Nonempty :=
  ⟨x, mem_pathComponent_self x⟩


theorem mem_pathComponent_of_mem (h : x ∈ pathComponent y) : y ∈ pathComponent x :=
  Joined.symm h


theorem pathComponent_symm : x ∈ pathComponent y ↔ y ∈ pathComponent x :=
  ⟨fun h => mem_pathComponent_of_mem h, fun h => mem_pathComponent_of_mem h⟩


theorem pathComponent_congr (h : x ∈ pathComponent y) : pathComponent x = pathComponent y := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    h : Membership.mem (pathComponent y) x
    ⊢ Eq (pathComponent x) (pathComponent y)
  -/
  ext z
  /-
    case h
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    h : Membership.mem (pathComponent y) x
    z : X
    ⊢ Iff (Membership.mem (pathComponent x) z) (Membership.mem (pathComponent y) z)
  -/
  constructor
    /-
      case h.mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y : X
      h : Membership.mem (pathComponent y) x
      z : X
      ⊢ Membership.mem (pathComponent x) z → Membership.mem (pathComponent y) z
    -/
  · intro h'
    /-
      case h.mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y : X
      h : Membership.mem (pathComponent y) x
      z : X
      h' : Membership.mem (pathComponent x) z
      ⊢ Membership.mem (pathComponent y) z
    -/
    rw [pathComponent_symm]
    /-
      case h.mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y : X
      h : Membership.mem (pathComponent y) x
      z : X
      h' : Membership.mem (pathComponent x) z
      ⊢ Membership.mem (pathComponent z) y
    -/
    exact (h.trans h').symm
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y : X
      h : Membership.mem (pathComponent y) x
      z : X
      ⊢ Membership.mem (pathComponent y) z → Membership.mem (pathComponent x) z
    -/
  · intro h'
    /-
      case h.mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y : X
      h : Membership.mem (pathComponent y) x
      z : X
      h' : Membership.mem (pathComponent y) z
      ⊢ Membership.mem (pathComponent x) z
    -/
    rw [pathComponent_symm] at h' ⊢
    /-
      case h.mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      x y : X
      h : Membership.mem (pathComponent y) x
      z : X
      h' : Membership.mem (pathComponent z) y
      ⊢ Membership.mem (pathComponent z) x
    -/
    exact h'.trans h
    /-
      🎉 no goals
    -/


theorem pathComponent_subset_component (x : X) : pathComponent x ⊆ connectedComponent x :=
  fun y h =>
                                                                             /-
                                                                               X : Type u_1
                                                                               inst✝ : TopologicalSpace X
                                                                               x y : X
                                                                               h : Membership.mem (pathComponent x) y
                                                                               ⊢ Eq ((Joined.somePath h) 0) x
                                                                             -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  (isConnected_range h.somePath.continuous).subset_connectedComponent ⟨0, by simp⟩ ⟨1, by simp⟩
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


/-- The path component of `x` in `F` is the set of points that can be joined to `x` in `F`. -/
def pathComponentIn (x : X) (F : Set X) :=
  { y | JoinedIn F x y }


@[simp]
theorem pathComponentIn_univ (x : X) : pathComponentIn x univ = pathComponent x := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x : X
    ⊢ Eq (pathComponentIn x Set.univ) (pathComponent x)
  -/
  simp [pathComponentIn, pathComponent, JoinedIn, Joined, exists_true_iff_nonempty]
  /-
    🎉 no goals
  -/


theorem Joined.mem_pathComponent (hyz : Joined y z) (hxy : y ∈ pathComponent x) :
    z ∈ pathComponent x :=
  hxy.trans hyz


theorem mem_pathComponentIn_self (h : x ∈ F) : x ∈ pathComponentIn x F :=
  JoinedIn.refl h


theorem pathComponentIn_subset : pathComponentIn x F ⊆ F :=
  fun _ hy ↦ hy.target_mem


theorem pathComponentIn_nonempty_iff : (pathComponentIn x F).Nonempty ↔ x ∈ F :=
  ⟨fun ⟨_, ⟨γ, hγ⟩⟩ ↦ γ.source ▸ hγ 0, fun hx ↦ ⟨x, mem_pathComponentIn_self hx⟩⟩


theorem pathComponentIn_congr (h : x ∈ pathComponentIn y F) :
    pathComponentIn x F = pathComponentIn y F := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : X
    F : Set X
    h : Membership.mem (pathComponentIn y F) x
    ⊢ Eq (pathComponentIn x F) (pathComponentIn y F)
  -/
  ext; exact ⟨h.trans, h.symm.trans⟩
       /-
         🎉 no goals
       -/


@[gcongr]
theorem pathComponentIn_mono {G : Set X} (h : F ⊆ G) :
    pathComponentIn x F ⊆ pathComponentIn x G :=
  fun _ ⟨γ, hγ⟩ ↦ ⟨γ, fun t ↦ h (hγ t)⟩


/-- A set `F` is path connected if it contains a point that can be joined to all other in `F`. -/
def IsPathConnected (F : Set X) : Prop :=
  ∃ x ∈ F, ∀ {y}, y ∈ F → JoinedIn F x y


theorem isPathConnected_iff_eq : IsPathConnected F ↔ ∃ x ∈ F, pathComponentIn x F = F := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    F : Set X
    ⊢ Iff (IsPathConnected F) (Exists fun x => And (Membership.mem F x) (Eq (pathC …
  -/
  constructor <;> rintro ⟨x, x_in, h⟩ <;> use x, x_in
    /-
      case right
      X : Type u_1
      inst✝ : TopologicalSpace X
      F : Set X
      x : X
      x_in : Membership.mem F x
      h : ∀ {y : X}, Membership.mem F y → JoinedIn F x y
      ⊢ Eq (pathComponentIn x F) F
    -/
  · ext y
    /-
      case right.h
      X : Type u_1
      inst✝ : TopologicalSpace X
      F : Set X
      x : X
      x_in : Membership.mem F x
      h : ∀ {y : X}, Membership.mem F y → JoinedIn F x y
      y : X
      ⊢ Iff (Membership.mem (pathComponentIn x F) y) (Membership.mem F y)
    -/
    exact ⟨fun hy => hy.mem.2, h⟩
    /-
      🎉 no goals
    -/
    /-
      case right
      X : Type u_1
      inst✝ : TopologicalSpace X
      F : Set X
      x : X
      x_in : Membership.mem F x
      h : Eq (pathComponentIn x F) F
      ⊢ ∀ {y : X}, Membership.mem F y → JoinedIn F x y
    -/
  · intro y y_in
    /-
      case right
      X : Type u_1
      inst✝ : TopologicalSpace X
      F : Set X
      x : X
      x_in : Membership.mem F x
      h : Eq (pathComponentIn x F) F
      y : X
      y_in : Membership.mem F y
      ⊢ JoinedIn F x y
    -/
    rwa [← h] at y_in
    /-
      🎉 no goals
    -/


theorem IsPathConnected.joinedIn (h : IsPathConnected F) :
    ∀ᵉ (x ∈ F) (y ∈ F), JoinedIn F x y := fun _x x_in _y y_in =>
  let ⟨_b, _b_in, hb⟩ := h
  (hb x_in).symm.trans (hb y_in)


theorem isPathConnected_iff :
    IsPathConnected F ↔ F.Nonempty ∧ ∀ᵉ (x ∈ F) (y ∈ F), JoinedIn F x y :=
  ⟨fun h =>
    ⟨let ⟨b, b_in, _hb⟩ := h; ⟨b, b_in⟩, h.joinedIn⟩,
    fun ⟨⟨b, b_in⟩, h⟩ => ⟨b, b_in, fun x_in => h _ b_in _ x_in⟩⟩


/-- If `f` is continuous on `F` and `F` is path-connected, so is `f(F)`. -/
theorem IsPathConnected.image' (hF : IsPathConnected F)
    {f : X → Y} (hf : ContinuousOn f F) : IsPathConnected (f '' F) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    F : Set X
    hF : IsPathConnected F
    f : X → Y
    hf : ContinuousOn f F
    ⊢ IsPathConnected (Set.image f F)
  -/
  rcases hF with ⟨x, x_in, hx⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    F : Set X
    f : X → Y
    hf : ContinuousOn f F
    x : X
    x_in : Membership.mem F x
    hx : ∀ {y : X}, Membership.mem F y → JoinedIn F x y
    ⊢ IsPathConnected (Set.image f F)
  -/
  use f x, mem_image_of_mem f x_in
  /-
    case right
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    F : Set X
    f : X → Y
    hf : ContinuousOn f F
    x : X
    x_in : Membership.mem F x
    hx : ∀ {y : X}, Membership.mem F y → JoinedIn F x y
    ⊢ ∀ {y : Y}, Membership.mem (Set.image f F) y → JoinedIn (Set.image f F) (f x) y
  -/
  rintro _ ⟨y, y_in, rfl⟩
  /-
    case right.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    F : Set X
    f : X → Y
    hf : ContinuousOn f F
    x : X
    x_in : Membership.mem F x
    hx : ∀ {y : X}, Membership.mem F y → JoinedIn F x y
    y : X
    y_in : Membership.mem F y
    ⊢ JoinedIn (Set.image f F) (f x) (f y)
  -/
  refine ⟨(hx y_in).somePath.map' ?_, fun t ↦ ⟨_, (hx y_in).somePath_mem t, rfl⟩⟩
  /-
    case right.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    F : Set X
    f : X → Y
    hf : ContinuousOn f F
    x : X
    x_in : Membership.mem F x
    hx : ∀ {y : X}, Membership.mem F y → JoinedIn F x y
    y : X
    y_in : Membership.mem F y
    ⊢ ContinuousOn f (Set.range ⇑⋯.somePath)
  -/
  exact hf.mono (range_subset_iff.2 (hx y_in).somePath_mem)
  /-
    🎉 no goals
  -/


/-- If `f` is continuous and `F` is path-connected, so is `f(F)`. -/
theorem IsPathConnected.image (hF : IsPathConnected F) {f : X → Y} (hf : Continuous f) :
    IsPathConnected (f '' F) :=
  hF.image' hf.continuousOn


/-- If `f : X → Y` is an inducing map, `f(F)` is path-connected iff `F` is. -/
nonrec theorem Topology.IsInducing.isPathConnected_iff {f : X → Y} (hf : IsInducing f) :
    IsPathConnected F ↔ IsPathConnected (f '' F) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    F : Set X
    f : X → Y
    hf : Topology.IsInducing f
    ⊢ Iff (IsPathConnected F) (IsPathConnected (Set.image f F))
  -/
  simp only [IsPathConnected, forall_mem_image, exists_mem_image]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    F : Set X
    f : X → Y
    hf : Topology.IsInducing f
    ⊢ Iff (Exists fun x => And (Membership.mem F x) (∀ {y : X}, Membership.mem F y …
  -/
  refine exists_congr fun x ↦ and_congr_right fun hx ↦ forall₂_congr fun y hy ↦ ?_
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    F : Set X
    f : X → Y
    hf : Topology.IsInducing f
    x : X
    hx : Membership.mem F x
    y : X
    hy : Membership.mem F y
    ⊢ Iff (JoinedIn F x y) (JoinedIn (Set.image f F) (f x) (f y))
  -/
  rw [hf.joinedIn_image hx hy]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias Inducing.isPathConnected_iff := IsInducing.isPathConnected_iff


/-- If `h : X → Y` is a homeomorphism, `h(s)` is path-connected iff `s` is. -/
@[simp]
theorem Homeomorph.isPathConnected_image {s : Set X} (h : X ≃ₜ Y) :
    IsPathConnected (h '' s) ↔ IsPathConnected s :=
  h.isInducing.isPathConnected_iff.symm


/-- If `h : X → Y` is a homeomorphism, `h⁻¹(s)` is path-connected iff `s` is. -/
@[simp]
theorem Homeomorph.isPathConnected_preimage {s : Set Y} (h : X ≃ₜ Y) :
    IsPathConnected (h ⁻¹' s) ↔ IsPathConnected s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set Y
    h : Homeomorph X Y
    ⊢ Iff (IsPathConnected (Set.preimage (⇑h) s)) (IsPathConnected s)
  -/
  rw [← Homeomorph.image_symm]; exact h.symm.isPathConnected_image
                                /-
                                  🎉 no goals
                                -/


theorem IsPathConnected.mem_pathComponent (h : IsPathConnected F) (x_in : x ∈ F) (y_in : y ∈ F) :
    y ∈ pathComponent x :=
  (h.joinedIn x x_in y y_in).joined


theorem IsPathConnected.subset_pathComponent (h : IsPathConnected F) (x_in : x ∈ F) :
    F ⊆ pathComponent x := fun _y y_in => h.mem_pathComponent x_in y_in


theorem IsPathConnected.subset_pathComponentIn {s : Set X} (hs : IsPathConnected s)
    (hxs : x ∈ s) (hsF : s ⊆ F) : s ⊆ pathComponentIn x F :=
  fun y hys ↦ (hs.joinedIn x hxs y hys).mono hsF


theorem isPathConnected_singleton (x : X) : IsPathConnected ({x} : Set X) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x : X
    ⊢ IsPathConnected (Singleton.singleton x)
  -/
  refine ⟨x, rfl, ?_⟩
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x : X
    ⊢ ∀ {y : X}, Membership.mem (Singleton.singleton x) y → JoinedIn (Singleton.si …
  -/
  rintro y rfl
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    y : X
    ⊢ JoinedIn (Singleton.singleton y) y y
  -/
  exact JoinedIn.refl rfl
  /-
    🎉 no goals
  -/


theorem isPathConnected_pathComponentIn (h : x ∈ F) : IsPathConnected (pathComponentIn x F) :=
  ⟨x, mem_pathComponentIn_self h, fun ⟨γ, hγ⟩ ↦ by
    refine ⟨γ, fun t ↦
      ⟨(γ.truncateOfLE t.2.1).cast (γ.extend_zero.symm) (γ.extend_extends' t).symm, fun t' ↦ ?_⟩⟩
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      x : X
      F : Set X
      h : Membership.mem F x
      y✝ : X
      x✝ : Membership.mem (pathComponentIn x F) y✝
      γ : Path x y✝
      hγ : ∀ (t : ↑unitInterval), Membership.mem F (γ t)
      t t' : ↑unitInterval
      ⊢ Membership.mem F (((γ.truncateOfLE ⋯).cast ⋯ ⋯) t')
    -/
    dsimp [Path.truncateOfLE, Path.truncate]
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      x : X
      F : Set X
      h : Membership.mem F x
      y✝ : X
      x✝ : Membership.mem (pathComponentIn x F) y✝
      γ : Path x y✝
      hγ : ∀ (t : ↑unitInterval), Membership.mem F (γ t)
      t t' : ↑unitInterval
      ⊢ Membership.mem F (γ.extend (Min.min (Max.max (↑t') 0) ↑t))
    -/
    exact γ.extend_extends' ⟨min (max t'.1 0) t.1, by simp [t.2.1, t.2.2]⟩ ▸ hγ _⟩
    /-
      🎉 no goals
    -/


theorem isPathConnected_pathComponent : IsPathConnected (pathComponent x) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x : X
    ⊢ IsPathConnected (pathComponent x)
  -/
  rw [← pathComponentIn_univ]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x : X
    ⊢ IsPathConnected (pathComponentIn x Set.univ)
  -/
  exact isPathConnected_pathComponentIn (mem_univ x)
  /-
    🎉 no goals
  -/


theorem IsPathConnected.union {U V : Set X} (hU : IsPathConnected U) (hV : IsPathConnected V)
    (hUV : (U ∩ V).Nonempty) : IsPathConnected (U ∪ V) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    U V : Set X
    hU : IsPathConnected U
    hV : IsPathConnected V
    hUV : (Inter.inter U V).Nonempty
    ⊢ IsPathConnected (Union.union U V)
  -/
  rcases hUV with ⟨x, xU, xV⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    U V : Set X
    hU : IsPathConnected U
    hV : IsPathConnected V
    x : X
    xU : Membership.mem U x
    xV : Membership.mem V x
    ⊢ IsPathConnected (Union.union U V)
  -/
  use x, Or.inl xU
  /-
    case right
    X : Type u_1
    inst✝ : TopologicalSpace X
    U V : Set X
    hU : IsPathConnected U
    hV : IsPathConnected V
    x : X
    xU : Membership.mem U x
    xV : Membership.mem V x
    ⊢ ∀ {y : X}, Membership.mem (Union.union U V) y → JoinedIn (Union.union U V) x y
  -/
  rintro y (yU | yV)
    /-
      case right.inl
      X : Type u_1
      inst✝ : TopologicalSpace X
      U V : Set X
      hU : IsPathConnected U
      hV : IsPathConnected V
      x : X
      xU : Membership.mem U x
      xV : Membership.mem V x
      y : X
      yU : Membership.mem U y
      ⊢ JoinedIn (Union.union U V) x y
    -/
  · exact (hU.joinedIn x xU y yU).mono subset_union_left
    /-
      🎉 no goals
    -/
    /-
      case right.inr
      X : Type u_1
      inst✝ : TopologicalSpace X
      U V : Set X
      hU : IsPathConnected U
      hV : IsPathConnected V
      x : X
      xU : Membership.mem U x
      xV : Membership.mem V x
      y : X
      yV : Membership.mem V y
      ⊢ JoinedIn (Union.union U V) x y
    -/
  · exact (hV.joinedIn x xV y yV).mono subset_union_right
    /-
      🎉 no goals
    -/


/-- If a set `W` is path-connected, then it is also path-connected when seen as a set in a smaller
ambient type `U` (when `U` contains `W`). -/
theorem IsPathConnected.preimage_coe {U W : Set X} (hW : IsPathConnected W) (hWU : W ⊆ U) :
    IsPathConnected (((↑) : U → X) ⁻¹' W) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    U W : Set X
    hW : IsPathConnected W
    hWU : HasSubset.Subset W U
    ⊢ IsPathConnected (Set.preimage Subtype.val W)
  -/
  rwa [IsInducing.subtypeVal.isPathConnected_iff, Subtype.image_preimage_val, inter_eq_right.2 hWU]
  /-
    🎉 no goals
  -/


theorem IsPathConnected.exists_path_through_family {n : ℕ}
    {s : Set X} (h : IsPathConnected s) (p : Fin (n + 1) → X) (hp : ∀ i, p i ∈ s) :
    ∃ γ : Path (p 0) (p n), range γ ⊆ s ∧ ∀ i, p i ∈ range γ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    ⊢ Exists fun γ => And (HasSubset.Subset (Set.range ⇑γ) s) (∀ (i : Fin (HAdd.hA …
  -/
  let p' : ℕ → X := fun k => if h : k < n + 1 then p ⟨k, h⟩ else p ⟨0, n.zero_lt_succ⟩
  obtain ⟨γ, hγ⟩ : ∃ γ : Path (p' 0) (p' n), (∀ i ≤ n, p' i ∈ range γ) ∧ range γ ⊆ s := by
    have hp' : ∀ i ≤ n, p' i ∈ s := by
      intro i hi
      simp [p', Nat.lt_succ_of_le hi, hp]
    clear_value p'
    clear hp p
    induction' n with n hn
    · use Path.refl (p' 0)
      constructor
      · rintro i hi
        rw [Nat.le_zero.mp hi]
        exact ⟨0, rfl⟩
      · rw [range_subset_iff]
        rintro _x
        exact hp' 0 le_rfl
    · rcases hn fun i hi => hp' i <| Nat.le_succ_of_le hi with ⟨γ₀, hγ₀⟩
      rcases h.joinedIn (p' n) (hp' n n.le_succ) (p' <| n + 1) (hp' (n + 1) <| le_rfl) with
        ⟨γ₁, hγ₁⟩
      let γ : Path (p' 0) (p' <| n + 1) := γ₀.trans γ₁
      use γ
      have range_eq : range γ = range γ₀ ∪ range γ₁ := γ₀.trans_range γ₁
      constructor
      · rintro i hi
        by_cases hi' : i ≤ n
        · rw [range_eq]
          left
          exact hγ₀.1 i hi'
        · rw [not_le, ← Nat.succ_le_iff] at hi'
          have : i = n.succ := le_antisymm hi hi'
          rw [this]
          use 1
          exact γ.target
      · rw [range_eq]
        apply union_subset hγ₀.2
        rw [range_subset_iff]
        exact hγ₁
  have hpp' : ∀ k < n + 1, p k = p' k := by
    intro k hk
    simp only [p', hk, dif_pos]
    congr
    ext
    rw [Fin.val_cast_of_lt hk]
  /-
    case intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    p' : Nat → X := fun k => dite (LT.lt k (HAdd.hAdd n 1)) (fun h => p ⟨k, h⟩) fu …
    γ : Path (p' 0) (p' n)
    hγ : And (∀ (i : Nat), LE.le i n → Membership.mem (Set.range ⇑γ) (p' i)) (HasS …
    hpp' : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (p ↑k) (p' k)
    ⊢ Exists fun γ => And (HasSubset.Subset (Set.range ⇑γ) s) (∀ (i : Fin (HAdd.hA …
  -/
  use γ.cast (hpp' 0 n.zero_lt_succ) (hpp' n n.lt_succ_self)
  /-
    case h
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    p' : Nat → X := fun k => dite (LT.lt k (HAdd.hAdd n 1)) (fun h => p ⟨k, h⟩) fu …
    γ : Path (p' 0) (p' n)
    hγ : And (∀ (i : Nat), LE.le i n → Membership.mem (Set.range ⇑γ) (p' i)) (HasS …
    hpp' : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (p ↑k) (p' k)
    ⊢ And (HasSubset.Subset (Set.range ⇑(γ.cast ⋯ ⋯)) s) (∀ (i : Fin (HAdd.hAdd n  …
  -/
  simp only [γ.cast_coe]
  /-
    case h
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    p' : Nat → X := fun k => dite (LT.lt k (HAdd.hAdd n 1)) (fun h => p ⟨k, h⟩) fu …
    γ : Path (p' 0) (p' n)
    hγ : And (∀ (i : Nat), LE.le i n → Membership.mem (Set.range ⇑γ) (p' i)) (HasS …
    hpp' : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (p ↑k) (p' k)
    ⊢ And (HasSubset.Subset (Set.range ⇑γ) s) (∀ (i : Fin (HAdd.hAdd n 1)), Member …
  -/
  refine And.intro hγ.2 ?_
  /-
    case h
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    p' : Nat → X := fun k => dite (LT.lt k (HAdd.hAdd n 1)) (fun h => p ⟨k, h⟩) fu …
    γ : Path (p' 0) (p' n)
    hγ : And (∀ (i : Nat), LE.le i n → Membership.mem (Set.range ⇑γ) (p' i)) (HasS …
    hpp' : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (p ↑k) (p' k)
    ⊢ ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem (Set.range ⇑γ) (p i)
  -/
  rintro ⟨i, hi⟩
  /-
    case h.mk
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    p' : Nat → X := fun k => dite (LT.lt k (HAdd.hAdd n 1)) (fun h => p ⟨k, h⟩) fu …
    γ : Path (p' 0) (p' n)
    hγ : And (∀ (i : Nat), LE.le i n → Membership.mem (Set.range ⇑γ) (p' i)) (HasS …
    hpp' : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (p ↑k) (p' k)
    i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    ⊢ Membership.mem (Set.range ⇑γ) (p ⟨i, hi⟩)
  -/
  suffices p ⟨i, hi⟩ = p' i by convert hγ.1 i (Nat.le_of_lt_succ hi)
  /-
    case h.mk
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    p' : Nat → X := fun k => dite (LT.lt k (HAdd.hAdd n 1)) (fun h => p ⟨k, h⟩) fu …
    γ : Path (p' 0) (p' n)
    hγ : And (∀ (i : Nat), LE.le i n → Membership.mem (Set.range ⇑γ) (p' i)) (HasS …
    hpp' : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (p ↑k) (p' k)
    i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    ⊢ Eq (p ⟨i, hi⟩) (p' i)
  -/
  rw [← hpp' i hi]
  /-
    case h.mk
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    p' : Nat → X := fun k => dite (LT.lt k (HAdd.hAdd n 1)) (fun h => p ⟨k, h⟩) fu …
    γ : Path (p' 0) (p' n)
    hγ : And (∀ (i : Nat), LE.le i n → Membership.mem (Set.range ⇑γ) (p' i)) (HasS …
    hpp' : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (p ↑k) (p' k)
    i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    ⊢ Eq (p ⟨i, hi⟩) (p ↑i)
  -/
  suffices i = i % n.succ by congr
  /-
    case h.mk
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    p' : Nat → X := fun k => dite (LT.lt k (HAdd.hAdd n 1)) (fun h => p ⟨k, h⟩) fu …
    γ : Path (p' 0) (p' n)
    hγ : And (∀ (i : Nat), LE.le i n → Membership.mem (Set.range ⇑γ) (p' i)) (HasS …
    hpp' : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (p ↑k) (p' k)
    i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    ⊢ Eq i (HMod.hMod i n.succ)
  -/
  rw [Nat.mod_eq_of_lt hi]
  /-
    🎉 no goals
  -/


theorem IsPathConnected.exists_path_through_family' {n : ℕ}
    {s : Set X} (h : IsPathConnected s) (p : Fin (n + 1) → X) (hp : ∀ i, p i ∈ s) :
    ∃ (γ : Path (p 0) (p n)) (t : Fin (n + 1) → I), (∀ t, γ t ∈ s) ∧ ∀ i, γ (t i) = p i := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    ⊢ Exists fun γ => Exists fun t => And (∀ (t : ↑unitInterval), Membership.mem s …
  -/
  rcases h.exists_path_through_family p hp with ⟨γ, hγ⟩
  /-
    case intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    γ : Path (p 0) (p ↑n)
    hγ : And (HasSubset.Subset (Set.range ⇑γ) s) (∀ (i : Fin (HAdd.hAdd n 1)), Mem …
    ⊢ Exists fun γ => Exists fun t => And (∀ (t : ↑unitInterval), Membership.mem s …
  -/
  rcases hγ with ⟨h₁, h₂⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    γ : Path (p 0) (p ↑n)
    h₁ : HasSubset.Subset (Set.range ⇑γ) s
    h₂ : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem (Set.range ⇑γ) (p i)
    ⊢ Exists fun γ => Exists fun t => And (∀ (t : ↑unitInterval), Membership.mem s …
  -/
  simp only [range, mem_setOf_eq] at h₂
  /-
    case intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    γ : Path (p 0) (p ↑n)
    h₁ : HasSubset.Subset (Set.range ⇑γ) s
    h₂ : ∀ (i : Fin (HAdd.hAdd n 1)), Exists fun y => Eq (γ y) (p i)
    ⊢ Exists fun γ => Exists fun t => And (∀ (t : ↑unitInterval), Membership.mem s …
  -/
  rw [range_subset_iff] at h₁
  /-
    case intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    γ : Path (p 0) (p ↑n)
    h₁ : ∀ (y : ↑unitInterval), Membership.mem s (γ y)
    h₂ : ∀ (i : Fin (HAdd.hAdd n 1)), Exists fun y => Eq (γ y) (p i)
    ⊢ Exists fun γ => Exists fun t => And (∀ (t : ↑unitInterval), Membership.mem s …
  -/
  choose! t ht using h₂
  /-
    case intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    n : Nat
    s : Set X
    h : IsPathConnected s
    p : Fin (HAdd.hAdd n 1) → X
    hp : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem s (p i)
    γ : Path (p 0) (p ↑n)
    h₁ : ∀ (y : ↑unitInterval), Membership.mem s (γ y)
    t : Fin (HAdd.hAdd n 1) → ↑unitInterval
    ht : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (γ (t i)) (p i)
    ⊢ Exists fun γ => Exists fun t => And (∀ (t : ↑unitInterval), Membership.mem s …
  -/
  exact ⟨γ, t, h₁, ht⟩
  /-
    🎉 no goals
  -/


/-- A topological space is path-connected if it is non-empty and every two points can be
joined by a continuous path. -/
@[mk_iff]
class PathConnectedSpace (X : Type*) [TopologicalSpace X] : Prop where
  /-- A path-connected space must be nonempty. -/
  nonempty : Nonempty X
  /-- Any two points in a path-connected space must be joined by a continuous path. -/
  joined : ∀ x y : X, Joined x y


theorem pathConnectedSpace_iff_zerothHomotopy :
    PathConnectedSpace X ↔ Nonempty (ZerothHomotopy X) ∧ Subsingleton (ZerothHomotopy X) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Iff (PathConnectedSpace X) (And (Nonempty (ZerothHomotopy X)) (Subsingleton  …
  -/
  letI := pathSetoid X
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    this : Setoid X := pathSetoid X
    ⊢ Iff (PathConnectedSpace X) (And (Nonempty (ZerothHomotopy X)) (Subsingleton  …
  -/
  constructor
    /-
      case mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      this : Setoid X := pathSetoid X
      ⊢ PathConnectedSpace X → And (Nonempty (ZerothHomotopy X)) (Subsingleton (Zero …
    -/
  · intro h
    /-
      case mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      this : Setoid X := pathSetoid X
      h : PathConnectedSpace X
      ⊢ And (Nonempty (ZerothHomotopy X)) (Subsingleton (ZerothHomotopy X))
    -/
    refine ⟨(nonempty_quotient_iff _).mpr h.1, ⟨?_⟩⟩
    /-
      case mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      this : Setoid X := pathSetoid X
      h : PathConnectedSpace X
      ⊢ ∀ (a b : ZerothHomotopy X), Eq a b
    -/
    rintro ⟨x⟩ ⟨y⟩
    /-
      case mp.mk.mk
      X : Type u_1
      inst✝ : TopologicalSpace X
      this : Setoid X := pathSetoid X
      h : PathConnectedSpace X
      a✝ : ZerothHomotopy X
      x : X
      b✝ : ZerothHomotopy X
      y : X
      ⊢ Eq (Quot.mk (⇑(pathSetoid X)) x) (Quot.mk (⇑(pathSetoid X)) y)
    -/
    exact Quotient.sound (PathConnectedSpace.joined x y)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      this : Setoid X := pathSetoid X
      ⊢ And (Nonempty (ZerothHomotopy X)) (Subsingleton (ZerothHomotopy X)) → PathCo …
    -/
  · unfold ZerothHomotopy
    /-
      case mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      this : Setoid X := pathSetoid X
      ⊢ And (Nonempty (Quotient (pathSetoid X))) (Subsingleton (Quotient (pathSetoid …
    -/
    rintro ⟨h, h'⟩
    /-
      case mpr.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      this : Setoid X := pathSetoid X
      h : Nonempty (Quotient (pathSetoid X))
      h' : Subsingleton (Quotient (pathSetoid X))
      ⊢ PathConnectedSpace X
    -/
    exact ⟨(nonempty_quotient_iff _).mp h, fun x y => Quotient.exact <| Subsingleton.elim ⟦x⟧ ⟦y⟧⟩
    /-
      🎉 no goals
    -/


/-- Use path-connectedness to build a path between two points. -/
def somePath (x y : X) : Path x y :=
  Nonempty.some (joined x y)


theorem pathConnectedSpace_iff_univ : PathConnectedSpace X ↔ IsPathConnected (univ : Set X) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Iff (PathConnectedSpace X) (IsPathConnected Set.univ)
  -/
  simp [pathConnectedSpace_iff, isPathConnected_iff, nonempty_iff_univ_nonempty]
  /-
    🎉 no goals
  -/


theorem isPathConnected_iff_pathConnectedSpace : IsPathConnected F ↔ PathConnectedSpace F := by
  rw [pathConnectedSpace_iff_univ, IsInducing.subtypeVal.isPathConnected_iff, image_univ,
    Subtype.range_val_subtype, setOf_mem_eq]


theorem isPathConnected_univ [PathConnectedSpace X] : IsPathConnected (univ : Set X) :=
  pathConnectedSpace_iff_univ.mp inferInstance


theorem isPathConnected_range [PathConnectedSpace X] {f : X → Y} (hf : Continuous f) :
    IsPathConnected (range f) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : PathConnectedSpace X
    f : X → Y
    hf : Continuous f
    ⊢ IsPathConnected (Set.range f)
  -/
  rw [← image_univ]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : PathConnectedSpace X
    f : X → Y
    hf : Continuous f
    ⊢ IsPathConnected (Set.image f Set.univ)
  -/
  exact isPathConnected_univ.image hf
  /-
    🎉 no goals
  -/


theorem Function.Surjective.pathConnectedSpace [PathConnectedSpace X]
    {f : X → Y} (hf : Surjective f) (hf' : Continuous f) : PathConnectedSpace Y := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : PathConnectedSpace X
    f : X → Y
    hf : Function.Surjective f
    hf' : Continuous f
    ⊢ PathConnectedSpace Y
  -/
  rw [pathConnectedSpace_iff_univ, ← hf.range_eq]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : PathConnectedSpace X
    f : X → Y
    hf : Function.Surjective f
    hf' : Continuous f
    ⊢ IsPathConnected (Set.range f)
  -/
  exact isPathConnected_range hf'
  /-
    🎉 no goals
  -/


instance Quotient.instPathConnectedSpace {s : Setoid X} [PathConnectedSpace X] :
    PathConnectedSpace (Quotient s) :=
  Quotient.mk'_surjective.pathConnectedSpace continuous_coinduced_rng


/-- This is a special case of `NormedSpace.instPathConnectedSpace` (and
`TopologicalAddGroup.pathConnectedSpace`). It exists only to simplify dependencies. -/
instance Real.instPathConnectedSpace : PathConnectedSpace ℝ where
                                                         /-
                                                           X : Type u_1
                                                           Y : Type u_2
                                                           inst✝¹ : TopologicalSpace X
                                                           inst✝ : TopologicalSpace Y
                                                           x✝ y✝ z : X
                                                           ι : Type u_3
                                                           F : Set X
                                                           x y : Real
                                                           ⊢ Continuous fun t => HAdd.hAdd (HMul.hMul (HSub.hSub 1 ↑t) x) (HMul.hMul (↑t) …
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  joined x y := ⟨⟨⟨fun (t : I) ↦ (1 - t) * x + t * y, by fun_prop⟩, by simp, by simp⟩⟩
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  nonempty := inferInstance


theorem pathConnectedSpace_iff_eq : PathConnectedSpace X ↔ ∃ x : X, pathComponent x = univ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Iff (PathConnectedSpace X) (Exists fun x => Eq (pathComponent x) Set.univ)
  -/
  simp [pathConnectedSpace_iff_univ, isPathConnected_iff_eq]
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 100) PathConnectedSpace.connectedSpace [PathConnectedSpace X] :
    ConnectedSpace X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    x y z : X
    ι : Type u_3
    F : Set X
    inst✝ : PathConnectedSpace X
    ⊢ ConnectedSpace X
  -/
  rw [connectedSpace_iff_connectedComponent]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    x y z : X
    ι : Type u_3
    F : Set X
    inst✝ : PathConnectedSpace X
    ⊢ Exists fun x => Eq (connectedComponent x) Set.univ
  -/
  rcases isPathConnected_iff_eq.mp (pathConnectedSpace_iff_univ.mp ‹_›) with ⟨x, _x_in, hx⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    x✝ y z : X
    ι : Type u_3
    F : Set X
    inst✝ : PathConnectedSpace X
    x : X
    _x_in : Membership.mem Set.univ x
    hx : Eq (pathComponentIn x Set.univ) Set.univ
    ⊢ Exists fun x => Eq (connectedComponent x) Set.univ
  -/
  use x
  /-
    case h
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    x✝ y z : X
    ι : Type u_3
    F : Set X
    inst✝ : PathConnectedSpace X
    x : X
    _x_in : Membership.mem Set.univ x
    hx : Eq (pathComponentIn x Set.univ) Set.univ
    ⊢ Eq (connectedComponent x) Set.univ
  -/
  rw [← univ_subset_iff]
  /-
    case h
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    x✝ y z : X
    ι : Type u_3
    F : Set X
    inst✝ : PathConnectedSpace X
    x : X
    _x_in : Membership.mem Set.univ x
    hx : Eq (pathComponentIn x Set.univ) Set.univ
    ⊢ HasSubset.Subset Set.univ (connectedComponent x)
  -/
  exact (by simpa using hx : pathComponent x = univ) ▸ pathComponent_subset_component x
  /-
    🎉 no goals
  -/


theorem IsPathConnected.isConnected (hF : IsPathConnected F) : IsConnected F := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    F : Set X
    hF : IsPathConnected F
    ⊢ IsConnected F
  -/
  rw [isConnected_iff_connectedSpace]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    F : Set X
    hF : IsPathConnected F
    ⊢ ConnectedSpace ↑F
  -/
  rw [isPathConnected_iff_pathConnectedSpace] at hF
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    F : Set X
    hF : PathConnectedSpace ↑F
    ⊢ ConnectedSpace ↑F
  -/
  exact @PathConnectedSpace.connectedSpace _ _ hF
  /-
    🎉 no goals
  -/


theorem exists_path_through_family {n : ℕ} (p : Fin (n + 1) → X) :
    ∃ γ : Path (p 0) (p n), ∀ i, p i ∈ range γ := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : PathConnectedSpace X
    n : Nat
    p : Fin (HAdd.hAdd n 1) → X
    ⊢ Exists fun γ => ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem (Set.range ⇑γ) ( …
  -/
  have : IsPathConnected (univ : Set X) := pathConnectedSpace_iff_univ.mp (by infer_instance)
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : PathConnectedSpace X
    n : Nat
    p : Fin (HAdd.hAdd n 1) → X
    this : IsPathConnected Set.univ
    ⊢ Exists fun γ => ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem (Set.range ⇑γ) ( …
  -/
  rcases this.exists_path_through_family p fun _i => True.intro with ⟨γ, -, h⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : PathConnectedSpace X
    n : Nat
    p : Fin (HAdd.hAdd n 1) → X
    this : IsPathConnected Set.univ
    γ : Path (p 0) (p ↑n)
    h : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem (Set.range ⇑γ) (p i)
    ⊢ Exists fun γ => ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem (Set.range ⇑γ) ( …
  -/
  exact ⟨γ, h⟩
  /-
    🎉 no goals
  -/


theorem exists_path_through_family' {n : ℕ} (p : Fin (n + 1) → X) :
    ∃ (γ : Path (p 0) (p n)) (t : Fin (n + 1) → I), ∀ i, γ (t i) = p i := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : PathConnectedSpace X
    n : Nat
    p : Fin (HAdd.hAdd n 1) → X
    ⊢ Exists fun γ => Exists fun t => ∀ (i : Fin (HAdd.hAdd n 1)), Eq (γ (t i)) (p …
  -/
  have : IsPathConnected (univ : Set X) := pathConnectedSpace_iff_univ.mp (by infer_instance)
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : PathConnectedSpace X
    n : Nat
    p : Fin (HAdd.hAdd n 1) → X
    this : IsPathConnected Set.univ
    ⊢ Exists fun γ => Exists fun t => ∀ (i : Fin (HAdd.hAdd n 1)), Eq (γ (t i)) (p …
  -/
  rcases this.exists_path_through_family' p fun _i => True.intro with ⟨γ, t, -, h⟩
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : PathConnectedSpace X
    n : Nat
    p : Fin (HAdd.hAdd n 1) → X
    this : IsPathConnected Set.univ
    γ : Path (p 0) (p ↑n)
    t : Fin (HAdd.hAdd n 1) → ↑unitInterval
    h : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (γ (t i)) (p i)
    ⊢ Exists fun γ => Exists fun t => ∀ (i : Fin (HAdd.hAdd n 1)), Eq (γ (t i)) (p …
  -/
  exact ⟨γ, t, h⟩
  /-
    🎉 no goals
  -/


/-- A topological space is locally path connected, at every point, path connected
neighborhoods form a neighborhood basis. -/
class LocPathConnectedSpace (X : Type*) [TopologicalSpace X] : Prop where
  /-- Each neighborhood filter has a basis of path-connected neighborhoods. -/
  path_connected_basis : ∀ x : X, (𝓝 x).HasBasis (fun s : Set X => s ∈ 𝓝 x ∧ IsPathConnected s) id


theorem LocPathConnectedSpace.of_bases {p : X → ι → Prop} {s : X → ι → Set X}
    (h : ∀ x, (𝓝 x).HasBasis (p x) (s x)) (h' : ∀ x i, p x i → IsPathConnected (s x i)) :
    LocPathConnectedSpace X where
  path_connected_basis x := by
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      ι : Type u_3
      p : X → ι → Prop
      s : X → ι → Set X
      h : ∀ (x : X), (nhds x).HasBasis (p x) (s x)
      h' : ∀ (x : X) (i : ι), p x i → IsPathConnected (s x i)
      x : X
      ⊢ (nhds x).HasBasis (fun s => And (Membership.mem (nhds x) s) (IsPathConnected …
    -/
    rw [hasBasis_self]
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      ι : Type u_3
      p : X → ι → Prop
      s : X → ι → Set X
      h : ∀ (x : X), (nhds x).HasBasis (p x) (s x)
      h' : ∀ (x : X) (i : ι), p x i → IsPathConnected (s x i)
      x : X
      ⊢ ∀ (t : Set X), Membership.mem (nhds x) t → Exists fun r => And (Membership.m …
    -/
    intro t ht
    /-
      X : Type u_1
      inst✝ : TopologicalSpace X
      ι : Type u_3
      p : X → ι → Prop
      s : X → ι → Set X
      h : ∀ (x : X), (nhds x).HasBasis (p x) (s x)
      h' : ∀ (x : X) (i : ι), p x i → IsPathConnected (s x i)
      x : X
      t : Set X
      ht : Membership.mem (nhds x) t
      ⊢ Exists fun r => And (Membership.mem (nhds x) r) (And (IsPathConnected r) (Ha …
    -/
    rcases (h x).mem_iff.mp ht with ⟨i, hpi, hi⟩
    /-
      case intro.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      ι : Type u_3
      p : X → ι → Prop
      s : X → ι → Set X
      h : ∀ (x : X), (nhds x).HasBasis (p x) (s x)
      h' : ∀ (x : X) (i : ι), p x i → IsPathConnected (s x i)
      x : X
      t : Set X
      ht : Membership.mem (nhds x) t
      i : ι
      hpi : p x i
      hi : HasSubset.Subset (s x i) t
      ⊢ Exists fun r => And (Membership.mem (nhds x) r) (And (IsPathConnected r) (Ha …
    -/
    exact ⟨s x i, (h x).mem_of_mem hpi, h' x i hpi, hi⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-16")]
alias locPathConnected_of_bases := LocPathConnectedSpace.of_bases


protected theorem IsOpen.pathComponentIn (x : X) (hF : IsOpen F) :
    IsOpen (pathComponentIn x F) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    F : Set X
    inst✝ : LocPathConnectedSpace X
    x : X
    hF : IsOpen F
    ⊢ IsOpen (pathComponentIn x F)
  -/
  rw [isOpen_iff_mem_nhds]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    F : Set X
    inst✝ : LocPathConnectedSpace X
    x : X
    hF : IsOpen F
    ⊢ ∀ (x_1 : X), Membership.mem (pathComponentIn x F) x_1 → Membership.mem (nhds …
  -/
  intro y hy
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    F : Set X
    inst✝ : LocPathConnectedSpace X
    x : X
    hF : IsOpen F
    y : X
    hy : Membership.mem (pathComponentIn x F) y
    ⊢ Membership.mem (nhds y) (pathComponentIn x F)
  -/
  let ⟨s, hs⟩ := (path_connected_basis y).mem_iff.mp (hF.mem_nhds (pathComponentIn_subset hy))
  exact mem_of_superset hs.1.1 <| pathComponentIn_congr hy ▸
    hs.1.2.subset_pathComponentIn (mem_of_mem_nhds hs.1.1) hs.2


/-- In a locally path connected space, each path component is an open set. -/
protected theorem IsOpen.pathComponent (x : X) : IsOpen (pathComponent x) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    x : X
    ⊢ IsOpen (pathComponent x)
  -/
  rw [← pathComponentIn_univ]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    x : X
    ⊢ IsOpen (pathComponentIn x Set.univ)
  -/
  exact isOpen_univ.pathComponentIn _
  /-
    🎉 no goals
  -/


/-- In a locally path connected space, each path component is a closed set. -/
protected theorem IsClosed.pathComponent (x : X) : IsClosed (pathComponent x) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    x : X
    ⊢ IsClosed (pathComponent x)
  -/
  rw [← isOpen_compl_iff, isOpen_iff_mem_nhds]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    x : X
    ⊢ ∀ (x_1 : X), Membership.mem (HasCompl.compl (pathComponent x)) x_1 → Members …
  -/
  intro y hxy
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    x y : X
    hxy : Membership.mem (HasCompl.compl (pathComponent x)) y
    ⊢ Membership.mem (nhds y) (HasCompl.compl (pathComponent x))
  -/
  rcases (path_connected_basis y).ex_mem with ⟨V, hVy, hVc⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    x y : X
    hxy : Membership.mem (HasCompl.compl (pathComponent x)) y
    V : Set X
    hVy : Membership.mem (nhds y) V
    hVc : IsPathConnected V
    ⊢ Membership.mem (nhds y) (HasCompl.compl (pathComponent x))
  -/
  filter_upwards [hVy] with z hz hxz
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    x y : X
    hxy : Membership.mem (HasCompl.compl (pathComponent x)) y
    V : Set X
    hVy : Membership.mem (nhds y) V
    hVc : IsPathConnected V
    z : X
    hz : Membership.mem V z
    hxz : Membership.mem (pathComponent x) z
    ⊢ False
  -/
  exact hxy <|  hxz.trans (hVc.joinedIn _ hz _ (mem_of_mem_nhds hVy)).joined
  /-
    🎉 no goals
  -/


/-- In a locally path connected space, each path component is a clopen set. -/
protected theorem IsClopen.pathComponent (x : X) : IsClopen (pathComponent x) :=
  ⟨.pathComponent x, .pathComponent x⟩


lemma pathComponentIn_mem_nhds (hF : F ∈ 𝓝 x) : pathComponentIn x F ∈ 𝓝 x := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    x : X
    F : Set X
    inst✝ : LocPathConnectedSpace X
    hF : Membership.mem (nhds x) F
    ⊢ Membership.mem (nhds x) (pathComponentIn x F)
  -/
  let ⟨u, huF, hu, hxu⟩ := mem_nhds_iff.mp hF
  exact mem_nhds_iff.mpr ⟨pathComponentIn x u, pathComponentIn_mono huF,
    hu.pathComponentIn x, mem_pathComponentIn_self hxu⟩


theorem pathConnectedSpace_iff_connectedSpace : PathConnectedSpace X ↔ ConnectedSpace X := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    ⊢ Iff (PathConnectedSpace X) (ConnectedSpace X)
  -/
  refine ⟨fun _ ↦ inferInstance, fun h ↦ ⟨inferInstance, fun x y ↦ ?_⟩⟩
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    h : ConnectedSpace X
    x y : X
    ⊢ Joined x y
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  rw [← mem_pathComponent_iff, (IsClopen.pathComponent _).eq_univ] <;> simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem pathComponent_eq_connectedComponent (x : X) : pathComponent x = connectedComponent x :=
  (pathComponent_subset_component x).antisymm <|
    (IsClopen.pathComponent x).connectedComponent_subset (mem_pathComponent_self _)


theorem pathConnected_subset_basis {U : Set X} (h : IsOpen U) (hx : x ∈ U) :
    (𝓝 x).HasBasis (fun s : Set X => s ∈ 𝓝 x ∧ IsPathConnected s ∧ s ⊆ U) id :=
  (path_connected_basis x).hasBasis_self_subset (IsOpen.mem_nhds h hx)


theorem isOpen_isPathConnected_basis (x : X) :
    (𝓝 x).HasBasis (fun s : Set X ↦ IsOpen s ∧ x ∈ s ∧ IsPathConnected s) id := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    x : X
    ⊢ (nhds x).HasBasis (fun s => And (IsOpen s) (And (Membership.mem s x) (IsPath …
  -/
  refine ⟨fun s ↦ ⟨fun hs ↦ ?_, fun ⟨u, hu⟩ ↦ mem_nhds_iff.mpr ⟨u, hu.2, hu.1.1, hu.1.2.1⟩⟩⟩
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    x : X
    s : Set X
    hs : Membership.mem (nhds x) s
    ⊢ Exists fun i => And (And (IsOpen i) (And (Membership.mem i x) (IsPathConnect …
  -/
  have ⟨u, hus, hu, hxu⟩ := mem_nhds_iff.mp hs
  exact ⟨pathComponentIn x u, ⟨hu.pathComponentIn _, ⟨mem_pathComponentIn_self hxu,
    isPathConnected_pathComponentIn hxu⟩⟩, pathComponentIn_subset.trans hus⟩


theorem Topology.IsOpenEmbedding.locPathConnectedSpace {e : Y → X} (he : IsOpenEmbedding e) :
    LocPathConnectedSpace Y :=
  have (y : Y) :
      (𝓝 y).HasBasis (fun s ↦ s ∈ 𝓝 (e y) ∧ IsPathConnected s ∧ s ⊆ range e) (e ⁻¹' ·) :=
    he.basis_nhds <| pathConnected_subset_basis he.isOpen_range (mem_range_self _)
  .of_bases this fun x s ⟨_, hs, hse⟩ ↦ by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : LocPathConnectedSpace X
      e : Y → X
      he : Topology.IsOpenEmbedding e
      this : ∀ (y : Y), (nhds y).HasBasis (fun s => And (Membership.mem (nhds (e y)) …
      x : Y
      s : Set X
      x✝ : And (Membership.mem (nhds (e x)) s) (And (IsPathConnected s) (HasSubset.S …
      left✝ : Membership.mem (nhds (e x)) s
      hs : IsPathConnected s
      hse : HasSubset.Subset s (Set.range e)
      ⊢ IsPathConnected (Set.preimage e s)
    -/
    rwa [he.isPathConnected_iff, image_preimage_eq_of_subset hse]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.locPathConnectedSpace := IsOpenEmbedding.locPathConnectedSpace


theorem IsOpen.locPathConnectedSpace {U : Set X} (h : IsOpen U) : LocPathConnectedSpace U :=
  h.isOpenEmbedding_subtypeVal.locPathConnectedSpace


@[deprecated (since := "2024-10-17")]
alias locPathConnected_of_isOpen := IsOpen.locPathConnectedSpace


theorem IsOpen.isConnected_iff_isPathConnected {U : Set X} (U_op : IsOpen U) :
    IsConnected U ↔ IsPathConnected U := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    U : Set X
    U_op : IsOpen U
    ⊢ Iff (IsConnected U) (IsPathConnected U)
  -/
  rw [isConnected_iff_connectedSpace, isPathConnected_iff_pathConnectedSpace]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    U : Set X
    U_op : IsOpen U
    ⊢ Iff (ConnectedSpace ↑U) (PathConnectedSpace ↑U)
  -/
  haveI := U_op.locPathConnectedSpace
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    U : Set X
    U_op : IsOpen U
    this : LocPathConnectedSpace ↑U
    ⊢ Iff (ConnectedSpace ↑U) (PathConnectedSpace ↑U)
  -/
  exact pathConnectedSpace_iff_connectedSpace.symm
  /-
    🎉 no goals
  -/


/-- Locally path-connected spaces are locally connected. -/
instance : LocallyConnectedSpace X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    x y z : X
    ι : Type u_3
    F : Set X
    inst✝ : LocPathConnectedSpace X
    ⊢ LocallyConnectedSpace X
  -/
  refine ⟨forall_imp (fun x h ↦ ⟨fun s ↦ ?_⟩) isOpen_isPathConnected_basis⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    x✝ y z : X
    ι : Type u_3
    F : Set X
    inst✝ : LocPathConnectedSpace X
    x : X
    h : (nhds x).HasBasis (fun s => And (IsOpen s) (And (Membership.mem s x) (IsPa …
    s : Set X
    ⊢ Iff (Membership.mem (nhds x) s) (Exists fun i => And (And (IsOpen i) (And (M …
  -/
  refine ⟨fun hs ↦ ?_, fun ⟨u, ⟨hu, hxu, _⟩, hus⟩ ↦ mem_nhds_iff.mpr ⟨u, hus, hu, hxu⟩⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    x✝ y z : X
    ι : Type u_3
    F : Set X
    inst✝ : LocPathConnectedSpace X
    x : X
    h : (nhds x).HasBasis (fun s => And (IsOpen s) (And (Membership.mem s x) (IsPa …
    s : Set X
    hs : Membership.mem (nhds x) s
    ⊢ Exists fun i => And (And (IsOpen i) (And (Membership.mem i x) (IsConnected i …
  -/
  let ⟨u, ⟨hu, hxu, hu'⟩, hus⟩ := (h.mem_iff' s).mp hs
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    x✝ y z : X
    ι : Type u_3
    F : Set X
    inst✝ : LocPathConnectedSpace X
    x : X
    h : (nhds x).HasBasis (fun s => And (IsOpen s) (And (Membership.mem s x) (IsPa …
    s : Set X
    hs : Membership.mem (nhds x) s
    u : Set X
    hu : IsOpen u
    hxu : Membership.mem u x
    hu' : IsPathConnected u
    hus : HasSubset.Subset (id u) s
    ⊢ Exists fun i => And (And (IsOpen i) (And (Membership.mem i x) (IsConnected i …
  -/
  exact ⟨u, ⟨hu, hxu, hu'.isConnected⟩, hus⟩
  /-
    🎉 no goals
  -/


/-- A space is locally path-connected iff all path components of open subsets are open. -/
lemma locPathConnectedSpace_iff_isOpen_pathComponentIn {X : Type*} [TopologicalSpace X] :
    LocPathConnectedSpace X ↔ ∀ (x : X) (u : Set X), IsOpen u → IsOpen (pathComponentIn x u) :=
  ⟨fun _ _ _ hu ↦ hu.pathComponentIn _, fun h ↦ ⟨fun x ↦ ⟨fun s ↦ by
    /-
      X : Type u_4
      inst✝ : TopologicalSpace X
      h : ∀ (x : X) (u : Set X), IsOpen u → IsOpen (pathComponentIn x u)
      x : X
      s : Set X
      ⊢ Iff (Membership.mem (nhds x) s) (Exists fun i => And (And (Membership.mem (n …
    -/
    refine ⟨fun hs ↦ ?_, fun ⟨_, ht⟩ ↦ Filter.mem_of_superset ht.1.1 ht.2⟩
    /-
      X : Type u_4
      inst✝ : TopologicalSpace X
      h : ∀ (x : X) (u : Set X), IsOpen u → IsOpen (pathComponentIn x u)
      x : X
      s : Set X
      hs : Membership.mem (nhds x) s
      ⊢ Exists fun i => And (And (Membership.mem (nhds x) i) (IsPathConnected i)) (H …
    -/
    let ⟨u, hu⟩ := mem_nhds_iff.mp hs
    exact ⟨pathComponentIn x u, ⟨(h x u hu.2.1).mem_nhds (mem_pathComponentIn_self hu.2.2),
      isPathConnected_pathComponentIn hu.2.2⟩, pathComponentIn_subset.trans hu.1⟩⟩⟩⟩


/-- A space is locally path-connected iff all path components of open subsets are neighbourhoods. -/
lemma locPathConnectedSpace_iff_pathComponentIn_mem_nhds {X : Type*} [TopologicalSpace X] :
    LocPathConnectedSpace X ↔
    ∀ x : X, ∀ u : Set X, IsOpen u → x ∈ u → pathComponentIn x u ∈ nhds x := by
  /-
    X : Type u_4
    inst✝ : TopologicalSpace X
    ⊢ Iff (LocPathConnectedSpace X) (∀ (x : X) (u : Set X), IsOpen u → Membership. …
  -/
  rw [locPathConnectedSpace_iff_isOpen_pathComponentIn]
  /-
    X : Type u_4
    inst✝ : TopologicalSpace X
    ⊢ Iff (∀ (x : X) (u : Set X), IsOpen u → IsOpen (pathComponentIn x u)) (∀ (x : …
  -/
  simp_rw [forall_comm (β := Set X), ← imp_forall_iff]
  /-
    X : Type u_4
    inst✝ : TopologicalSpace X
    ⊢ Iff (∀ (b : Set X), IsOpen b → ∀ (x : X), IsOpen (pathComponentIn x b)) (∀ ( …
  -/
  refine forall_congr' fun u ↦ imp_congr_right fun _ ↦ ?_
  exact ⟨fun h x hxu ↦ (h x).mem_nhds (mem_pathComponentIn_self hxu),
    fun h x ↦ isOpen_iff_mem_nhds.mpr fun y hy ↦
      pathComponentIn_congr hy ▸ h y <| pathComponentIn_subset hy⟩


/-- Any topology coinduced by a locally path-connected topology is locally path-connected. -/
lemma LocPathConnectedSpace.coinduced {Y : Type*} (f : X → Y) :
    @LocPathConnectedSpace Y (.coinduced f ‹_›) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    Y : Type u_4
    f : X → Y
    ⊢ LocPathConnectedSpace Y
  -/
  let _ := TopologicalSpace.coinduced f ‹_›; have hf : Continuous f := continuous_coinduced_rng
  refine locPathConnectedSpace_iff_isOpen_pathComponentIn.mpr fun y u hu ↦
    isOpen_coinduced.mpr <| isOpen_iff_mem_nhds.mpr fun x hx ↦ ?_
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    Y : Type u_4
    f : X → Y
    x✝ : TopologicalSpace Y := TopologicalSpace.coinduced f inst✝¹
    hf : Continuous f
    y : Y
    u : Set Y
    hu : IsOpen u
    x : X
    hx : Membership.mem (Set.preimage f (pathComponentIn y u)) x
    ⊢ Membership.mem (nhds x) (Set.preimage f (pathComponentIn y u))
  -/
  have hx' := preimage_mono pathComponentIn_subset hx
  refine mem_nhds_iff.mpr ⟨pathComponentIn x (f ⁻¹' u), ?_,
    (hu.preimage hf).pathComponentIn _, mem_pathComponentIn_self hx'⟩
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocPathConnectedSpace X
    Y : Type u_4
    f : X → Y
    x✝ : TopologicalSpace Y := TopologicalSpace.coinduced f inst✝¹
    hf : Continuous f
    y : Y
    u : Set Y
    hu : IsOpen u
    x : X
    hx : Membership.mem (Set.preimage f (pathComponentIn y u)) x
    hx' : Membership.mem (Set.preimage f u) x
    ⊢ HasSubset.Subset (pathComponentIn x (Set.preimage f u)) (Set.preimage f (pat …
  -/
  rw [← image_subset_iff, ← pathComponentIn_congr hx]
  exact ((isPathConnected_pathComponentIn hx').image hf).subset_pathComponentIn
    ⟨x, mem_pathComponentIn_self hx', rfl⟩ <|
    (image_mono pathComponentIn_subset).trans <| u.image_preimage_subset f


/-- Quotients of locally path-connected spaces are locally path-connected. -/
lemma Topology.IsQuotientMap.locPathConnectedSpace {f : X → Y} (h : IsQuotientMap f) :
    LocPathConnectedSpace Y :=
  h.2 ▸ LocPathConnectedSpace.coinduced f


/-- Quotients of locally path-connected spaces are locally path-connected. -/
instance Quot.locPathConnectedSpace {r : X → X → Prop} : LocPathConnectedSpace (Quot r) :=
  isQuotientMap_quot_mk.locPathConnectedSpace


/-- Quotients of locally path-connected spaces are locally path-connected. -/
instance Quotient.locPathConnectedSpace {s : Setoid X} : LocPathConnectedSpace (Quotient s) :=
  isQuotientMap_quotient_mk'.locPathConnectedSpace



/-- Disjoint unions of locally path-connected spaces are locally path-connected. -/
instance Sum.locPathConnectedSpace.{u} {X Y : Type u} [TopologicalSpace X] [TopologicalSpace Y]
    [LocPathConnectedSpace X] [LocPathConnectedSpace Y] :
    LocPathConnectedSpace (X ⊕ Y) := by
  /-
    X✝ : Type u_1
    Y✝ : Type u_2
    inst✝⁶ : TopologicalSpace X✝
    inst✝⁵ : TopologicalSpace Y✝
    x y z : X✝
    ι : Type u_3
    F : Set X✝
    inst✝⁴ : LocPathConnectedSpace X✝
    X Y : Type u
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : LocPathConnectedSpace X
    inst✝ : LocPathConnectedSpace Y
    ⊢ LocPathConnectedSpace (Sum X Y)
  -/
  rw [locPathConnectedSpace_iff_pathComponentIn_mem_nhds]; intro x u hu hxu; rw [mem_nhds_iff]
  /-
    X✝ : Type u_1
    Y✝ : Type u_2
    inst✝⁶ : TopologicalSpace X✝
    inst✝⁵ : TopologicalSpace Y✝
    x✝ y z : X✝
    ι : Type u_3
    F : Set X✝
    inst✝⁴ : LocPathConnectedSpace X✝
    X Y : Type u
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : LocPathConnectedSpace X
    inst✝ : LocPathConnectedSpace Y
    x : Sum X Y
    u : Set (Sum X Y)
    hu : IsOpen u
    hxu : Membership.mem u x
    ⊢ Exists fun t => And (HasSubset.Subset t (pathComponentIn x u)) (And (IsOpen  …
  -/
  obtain x | y := x
    /-
      case inl
      X✝ : Type u_1
      Y✝ : Type u_2
      inst✝⁶ : TopologicalSpace X✝
      inst✝⁵ : TopologicalSpace Y✝
      x✝ y z : X✝
      ι : Type u_3
      F : Set X✝
      inst✝⁴ : LocPathConnectedSpace X✝
      X Y : Type u
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : LocPathConnectedSpace X
      inst✝ : LocPathConnectedSpace Y
      u : Set (Sum X Y)
      hu : IsOpen u
      x : X
      hxu : Membership.mem u (Sum.inl x)
      ⊢ Exists fun t => And (HasSubset.Subset t (pathComponentIn (Sum.inl x) u)) (An …
    -/
  · refine ⟨Sum.inl '' (pathComponentIn x (Sum.inl ⁻¹' u)), ?_, ?_, ?_⟩
      /-
        case inl.refine_1
        X✝ : Type u_1
        Y✝ : Type u_2
        inst✝⁶ : TopologicalSpace X✝
        inst✝⁵ : TopologicalSpace Y✝
        x✝ y z : X✝
        ι : Type u_3
        F : Set X✝
        inst✝⁴ : LocPathConnectedSpace X✝
        X Y : Type u
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : LocPathConnectedSpace X
        inst✝ : LocPathConnectedSpace Y
        u : Set (Sum X Y)
        hu : IsOpen u
        x : X
        hxu : Membership.mem u (Sum.inl x)
        ⊢ HasSubset.Subset (Set.image Sum.inl (pathComponentIn x (Set.preimage Sum.inl …
      -/
    · apply IsPathConnected.subset_pathComponentIn
        /-
          case inl.refine_1.hs
          X✝ : Type u_1
          Y✝ : Type u_2
          inst✝⁶ : TopologicalSpace X✝
          inst✝⁵ : TopologicalSpace Y✝
          x✝ y z : X✝
          ι : Type u_3
          F : Set X✝
          inst✝⁴ : LocPathConnectedSpace X✝
          X Y : Type u
          inst✝³ : TopologicalSpace X
          inst✝² : TopologicalSpace Y
          inst✝¹ : LocPathConnectedSpace X
          inst✝ : LocPathConnectedSpace Y
          u : Set (Sum X Y)
          hu : IsOpen u
          x : X
          hxu : Membership.mem u (Sum.inl x)
          ⊢ IsPathConnected (Set.image Sum.inl (pathComponentIn x (Set.preimage Sum.inl  …
        -/
      · exact (isPathConnected_pathComponentIn (by exact hxu)).image continuous_inl
        /-
          🎉 no goals
        -/
        /-
          case inl.refine_1.hxs
          X✝ : Type u_1
          Y✝ : Type u_2
          inst✝⁶ : TopologicalSpace X✝
          inst✝⁵ : TopologicalSpace Y✝
          x✝ y z : X✝
          ι : Type u_3
          F : Set X✝
          inst✝⁴ : LocPathConnectedSpace X✝
          X Y : Type u
          inst✝³ : TopologicalSpace X
          inst✝² : TopologicalSpace Y
          inst✝¹ : LocPathConnectedSpace X
          inst✝ : LocPathConnectedSpace Y
          u : Set (Sum X Y)
          hu : IsOpen u
          x : X
          hxu : Membership.mem u (Sum.inl x)
          ⊢ Membership.mem (Set.image Sum.inl (pathComponentIn x (Set.preimage Sum.inl u …
        -/
      · exact ⟨x, mem_pathComponentIn_self hxu, rfl⟩
        /-
          🎉 no goals
        -/
        /-
          case inl.refine_1.hsF
          X✝ : Type u_1
          Y✝ : Type u_2
          inst✝⁶ : TopologicalSpace X✝
          inst✝⁵ : TopologicalSpace Y✝
          x✝ y z : X✝
          ι : Type u_3
          F : Set X✝
          inst✝⁴ : LocPathConnectedSpace X✝
          X Y : Type u
          inst✝³ : TopologicalSpace X
          inst✝² : TopologicalSpace Y
          inst✝¹ : LocPathConnectedSpace X
          inst✝ : LocPathConnectedSpace Y
          u : Set (Sum X Y)
          hu : IsOpen u
          x : X
          hxu : Membership.mem u (Sum.inl x)
          ⊢ HasSubset.Subset (Set.image Sum.inl (pathComponentIn x (Set.preimage Sum.inl …
        -/
      · exact (image_mono pathComponentIn_subset).trans (u.image_preimage_subset _)
        /-
          🎉 no goals
        -/
      /-
        case inl.refine_2
        X✝ : Type u_1
        Y✝ : Type u_2
        inst✝⁶ : TopologicalSpace X✝
        inst✝⁵ : TopologicalSpace Y✝
        x✝ y z : X✝
        ι : Type u_3
        F : Set X✝
        inst✝⁴ : LocPathConnectedSpace X✝
        X Y : Type u
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : LocPathConnectedSpace X
        inst✝ : LocPathConnectedSpace Y
        u : Set (Sum X Y)
        hu : IsOpen u
        x : X
        hxu : Membership.mem u (Sum.inl x)
        ⊢ IsOpen (Set.image Sum.inl (pathComponentIn x (Set.preimage Sum.inl u)))
      -/
    · exact isOpenMap_inl _ <| (hu.preimage continuous_inl).pathComponentIn _
      /-
        🎉 no goals
      -/
      /-
        case inl.refine_3
        X✝ : Type u_1
        Y✝ : Type u_2
        inst✝⁶ : TopologicalSpace X✝
        inst✝⁵ : TopologicalSpace Y✝
        x✝ y z : X✝
        ι : Type u_3
        F : Set X✝
        inst✝⁴ : LocPathConnectedSpace X✝
        X Y : Type u
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : LocPathConnectedSpace X
        inst✝ : LocPathConnectedSpace Y
        u : Set (Sum X Y)
        hu : IsOpen u
        x : X
        hxu : Membership.mem u (Sum.inl x)
        ⊢ Membership.mem (Set.image Sum.inl (pathComponentIn x (Set.preimage Sum.inl u …
      -/
    · exact ⟨x, mem_pathComponentIn_self hxu, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case inr
      X✝ : Type u_1
      Y✝ : Type u_2
      inst✝⁶ : TopologicalSpace X✝
      inst✝⁵ : TopologicalSpace Y✝
      x y✝ z : X✝
      ι : Type u_3
      F : Set X✝
      inst✝⁴ : LocPathConnectedSpace X✝
      X Y : Type u
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : LocPathConnectedSpace X
      inst✝ : LocPathConnectedSpace Y
      u : Set (Sum X Y)
      hu : IsOpen u
      y : Y
      hxu : Membership.mem u (Sum.inr y)
      ⊢ Exists fun t => And (HasSubset.Subset t (pathComponentIn (Sum.inr y) u)) (An …
    -/
  · refine ⟨Sum.inr '' (pathComponentIn y (Sum.inr ⁻¹' u)), ?_, ?_, ?_⟩
      /-
        case inr.refine_1
        X✝ : Type u_1
        Y✝ : Type u_2
        inst✝⁶ : TopologicalSpace X✝
        inst✝⁵ : TopologicalSpace Y✝
        x y✝ z : X✝
        ι : Type u_3
        F : Set X✝
        inst✝⁴ : LocPathConnectedSpace X✝
        X Y : Type u
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : LocPathConnectedSpace X
        inst✝ : LocPathConnectedSpace Y
        u : Set (Sum X Y)
        hu : IsOpen u
        y : Y
        hxu : Membership.mem u (Sum.inr y)
        ⊢ HasSubset.Subset (Set.image Sum.inr (pathComponentIn y (Set.preimage Sum.inr …
      -/
    · apply IsPathConnected.subset_pathComponentIn
        /-
          case inr.refine_1.hs
          X✝ : Type u_1
          Y✝ : Type u_2
          inst✝⁶ : TopologicalSpace X✝
          inst✝⁵ : TopologicalSpace Y✝
          x y✝ z : X✝
          ι : Type u_3
          F : Set X✝
          inst✝⁴ : LocPathConnectedSpace X✝
          X Y : Type u
          inst✝³ : TopologicalSpace X
          inst✝² : TopologicalSpace Y
          inst✝¹ : LocPathConnectedSpace X
          inst✝ : LocPathConnectedSpace Y
          u : Set (Sum X Y)
          hu : IsOpen u
          y : Y
          hxu : Membership.mem u (Sum.inr y)
          ⊢ IsPathConnected (Set.image Sum.inr (pathComponentIn y (Set.preimage Sum.inr  …
        -/
      · exact (isPathConnected_pathComponentIn (by exact hxu)).image continuous_inr
        /-
          🎉 no goals
        -/
        /-
          case inr.refine_1.hxs
          X✝ : Type u_1
          Y✝ : Type u_2
          inst✝⁶ : TopologicalSpace X✝
          inst✝⁵ : TopologicalSpace Y✝
          x y✝ z : X✝
          ι : Type u_3
          F : Set X✝
          inst✝⁴ : LocPathConnectedSpace X✝
          X Y : Type u
          inst✝³ : TopologicalSpace X
          inst✝² : TopologicalSpace Y
          inst✝¹ : LocPathConnectedSpace X
          inst✝ : LocPathConnectedSpace Y
          u : Set (Sum X Y)
          hu : IsOpen u
          y : Y
          hxu : Membership.mem u (Sum.inr y)
          ⊢ Membership.mem (Set.image Sum.inr (pathComponentIn y (Set.preimage Sum.inr u …
        -/
      · exact ⟨y, mem_pathComponentIn_self hxu, rfl⟩
        /-
          🎉 no goals
        -/
        /-
          case inr.refine_1.hsF
          X✝ : Type u_1
          Y✝ : Type u_2
          inst✝⁶ : TopologicalSpace X✝
          inst✝⁵ : TopologicalSpace Y✝
          x y✝ z : X✝
          ι : Type u_3
          F : Set X✝
          inst✝⁴ : LocPathConnectedSpace X✝
          X Y : Type u
          inst✝³ : TopologicalSpace X
          inst✝² : TopologicalSpace Y
          inst✝¹ : LocPathConnectedSpace X
          inst✝ : LocPathConnectedSpace Y
          u : Set (Sum X Y)
          hu : IsOpen u
          y : Y
          hxu : Membership.mem u (Sum.inr y)
          ⊢ HasSubset.Subset (Set.image Sum.inr (pathComponentIn y (Set.preimage Sum.inr …
        -/
      · exact (image_mono pathComponentIn_subset).trans (u.image_preimage_subset _)
        /-
          🎉 no goals
        -/
      /-
        case inr.refine_2
        X✝ : Type u_1
        Y✝ : Type u_2
        inst✝⁶ : TopologicalSpace X✝
        inst✝⁵ : TopologicalSpace Y✝
        x y✝ z : X✝
        ι : Type u_3
        F : Set X✝
        inst✝⁴ : LocPathConnectedSpace X✝
        X Y : Type u
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : LocPathConnectedSpace X
        inst✝ : LocPathConnectedSpace Y
        u : Set (Sum X Y)
        hu : IsOpen u
        y : Y
        hxu : Membership.mem u (Sum.inr y)
        ⊢ IsOpen (Set.image Sum.inr (pathComponentIn y (Set.preimage Sum.inr u)))
      -/
    · exact isOpenMap_inr _ <| (hu.preimage continuous_inr).pathComponentIn _
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_3
        X✝ : Type u_1
        Y✝ : Type u_2
        inst✝⁶ : TopologicalSpace X✝
        inst✝⁵ : TopologicalSpace Y✝
        x y✝ z : X✝
        ι : Type u_3
        F : Set X✝
        inst✝⁴ : LocPathConnectedSpace X✝
        X Y : Type u
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : LocPathConnectedSpace X
        inst✝ : LocPathConnectedSpace Y
        u : Set (Sum X Y)
        hu : IsOpen u
        y : Y
        hxu : Membership.mem u (Sum.inr y)
        ⊢ Membership.mem (Set.image Sum.inr (pathComponentIn y (Set.preimage Sum.inr u …
      -/
    · exact ⟨y, mem_pathComponentIn_self hxu, rfl⟩
      /-
        🎉 no goals
      -/



/-- Disjoint unions of locally path-connected spaces are locally path-connected. -/
instance Sigma.locPathConnectedSpace {X : ι → Type*}
    [(i : ι) → TopologicalSpace (X i)] [(i : ι) → LocPathConnectedSpace (X i)] :
    LocPathConnectedSpace ((i : ι) × X i) := by
  /-
    X✝ : Type u_1
    Y : Type u_2
    inst✝⁴ : TopologicalSpace X✝
    inst✝³ : TopologicalSpace Y
    x y z : X✝
    ι : Type u_3
    F : Set X✝
    inst✝² : LocPathConnectedSpace X✝
    X : ι → Type u_4
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : ∀ (i : ι), LocPathConnectedSpace (X i)
    ⊢ LocPathConnectedSpace (Sigma fun i => X i)
  -/
  rw [locPathConnectedSpace_iff_pathComponentIn_mem_nhds]; intro x u hu hxu; rw [mem_nhds_iff]
  /-
    X✝ : Type u_1
    Y : Type u_2
    inst✝⁴ : TopologicalSpace X✝
    inst✝³ : TopologicalSpace Y
    x✝ y z : X✝
    ι : Type u_3
    F : Set X✝
    inst✝² : LocPathConnectedSpace X✝
    X : ι → Type u_4
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : ∀ (i : ι), LocPathConnectedSpace (X i)
    x : Sigma fun i => X i
    u : Set (Sigma fun i => X i)
    hu : IsOpen u
    hxu : Membership.mem u x
    ⊢ Exists fun t => And (HasSubset.Subset t (pathComponentIn x u)) (And (IsOpen  …
  -/
  refine ⟨(Sigma.mk x.1) '' (pathComponentIn x.2 ((Sigma.mk x.1) ⁻¹' u)), ?_, ?_, ?_⟩
    /-
      case refine_1
      X✝ : Type u_1
      Y : Type u_2
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      x✝ y z : X✝
      ι : Type u_3
      F : Set X✝
      inst✝² : LocPathConnectedSpace X✝
      X : ι → Type u_4
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : ∀ (i : ι), LocPathConnectedSpace (X i)
      x : Sigma fun i => X i
      u : Set (Sigma fun i => X i)
      hu : IsOpen u
      hxu : Membership.mem u x
      ⊢ HasSubset.Subset (Set.image (Sigma.mk x.fst) (pathComponentIn x.snd (Set.pre …
    -/
  · apply IsPathConnected.subset_pathComponentIn
      /-
        case refine_1.hs
        X✝ : Type u_1
        Y : Type u_2
        inst✝⁴ : TopologicalSpace X✝
        inst✝³ : TopologicalSpace Y
        x✝ y z : X✝
        ι : Type u_3
        F : Set X✝
        inst✝² : LocPathConnectedSpace X✝
        X : ι → Type u_4
        inst✝¹ : (i : ι) → TopologicalSpace (X i)
        inst✝ : ∀ (i : ι), LocPathConnectedSpace (X i)
        x : Sigma fun i => X i
        u : Set (Sigma fun i => X i)
        hu : IsOpen u
        hxu : Membership.mem u x
        ⊢ IsPathConnected (Set.image (Sigma.mk x.fst) (pathComponentIn x.snd (Set.prei …
      -/
    · exact (isPathConnected_pathComponentIn (by exact hxu)).image continuous_sigmaMk
      /-
        🎉 no goals
      -/
      /-
        case refine_1.hxs
        X✝ : Type u_1
        Y : Type u_2
        inst✝⁴ : TopologicalSpace X✝
        inst✝³ : TopologicalSpace Y
        x✝ y z : X✝
        ι : Type u_3
        F : Set X✝
        inst✝² : LocPathConnectedSpace X✝
        X : ι → Type u_4
        inst✝¹ : (i : ι) → TopologicalSpace (X i)
        inst✝ : ∀ (i : ι), LocPathConnectedSpace (X i)
        x : Sigma fun i => X i
        u : Set (Sigma fun i => X i)
        hu : IsOpen u
        hxu : Membership.mem u x
        ⊢ Membership.mem (Set.image (Sigma.mk x.fst) (pathComponentIn x.snd (Set.preim …
      -/
    · exact ⟨x.2, mem_pathComponentIn_self hxu, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.hsF
        X✝ : Type u_1
        Y : Type u_2
        inst✝⁴ : TopologicalSpace X✝
        inst✝³ : TopologicalSpace Y
        x✝ y z : X✝
        ι : Type u_3
        F : Set X✝
        inst✝² : LocPathConnectedSpace X✝
        X : ι → Type u_4
        inst✝¹ : (i : ι) → TopologicalSpace (X i)
        inst✝ : ∀ (i : ι), LocPathConnectedSpace (X i)
        x : Sigma fun i => X i
        u : Set (Sigma fun i => X i)
        hu : IsOpen u
        hxu : Membership.mem u x
        ⊢ HasSubset.Subset (Set.image (Sigma.mk x.fst) (pathComponentIn x.snd (Set.pre …
      -/
    · exact (image_mono pathComponentIn_subset).trans (u.image_preimage_subset _)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      X✝ : Type u_1
      Y : Type u_2
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      x✝ y z : X✝
      ι : Type u_3
      F : Set X✝
      inst✝² : LocPathConnectedSpace X✝
      X : ι → Type u_4
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : ∀ (i : ι), LocPathConnectedSpace (X i)
      x : Sigma fun i => X i
      u : Set (Sigma fun i => X i)
      hu : IsOpen u
      hxu : Membership.mem u x
      ⊢ IsOpen (Set.image (Sigma.mk x.fst) (pathComponentIn x.snd (Set.preimage (Sig …
    -/
  · exact isOpenMap_sigmaMk _ <| (hu.preimage continuous_sigmaMk).pathComponentIn _
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      X✝ : Type u_1
      Y : Type u_2
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      x✝ y z : X✝
      ι : Type u_3
      F : Set X✝
      inst✝² : LocPathConnectedSpace X✝
      X : ι → Type u_4
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : ∀ (i : ι), LocPathConnectedSpace (X i)
      x : Sigma fun i => X i
      u : Set (Sigma fun i => X i)
      hu : IsOpen u
      hxu : Membership.mem u x
      ⊢ Membership.mem (Set.image (Sigma.mk x.fst) (pathComponentIn x.snd (Set.preim …
    -/
  · exact ⟨x.2, mem_pathComponentIn_self hxu, rfl⟩
    /-
      🎉 no goals
    -/


