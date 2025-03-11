/-- When `E` is a topological module over a topological ring `R`, the class `SeparatingDual R E`
registers that continuous linear forms on `E` separate points of `E`. -/
@[mk_iff separatingDual_def]
class SeparatingDual (R V : Type*) [Ring R] [AddCommGroup V] [TopologicalSpace V]
    [TopologicalSpace R] [Module R V] : Prop where
  /-- Any nonzero vector can be mapped by a continuous linear map to a nonzero scalar. -/
  exists_ne_zero' : ∀ (x : V), x ≠ 0 → ∃ f : V →L[R] R, f x ≠ 0


instance {E : Type*} [TopologicalSpace E] [AddCommGroup E] [TopologicalAddGroup E]
    [Module ℝ E] [ContinuousSMul ℝ E] [LocallyConvexSpace ℝ E] [T1Space E] : SeparatingDual ℝ E :=
  ⟨fun x hx ↦ by
    /-
      E : Type u_1
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalAddGroup E
      inst✝³ : Module Real E
      inst✝² : ContinuousSMul Real E
      inst✝¹ : LocallyConvexSpace Real E
      inst✝ : T1Space E
      x : E
      hx : Ne x 0
      ⊢ Exists fun f => Ne (f x) 0
    -/
    rcases geometric_hahn_banach_point_point hx.symm with ⟨f, hf⟩
    /-
      case intro
      E : Type u_1
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalAddGroup E
      inst✝³ : Module Real E
      inst✝² : ContinuousSMul Real E
      inst✝¹ : LocallyConvexSpace Real E
      inst✝ : T1Space E
      x : E
      hx : Ne x 0
      f : ContinuousLinearMap (RingHom.id Real) E Real
      hf : LT.lt (f 0) (f x)
      ⊢ Exists fun f => Ne (f x) 0
    -/
    simp only [map_zero] at hf
    /-
      case intro
      E : Type u_1
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalAddGroup E
      inst✝³ : Module Real E
      inst✝² : ContinuousSMul Real E
      inst✝¹ : LocallyConvexSpace Real E
      inst✝ : T1Space E
      x : E
      hx : Ne x 0
      f : ContinuousLinearMap (RingHom.id Real) E Real
      hf : LT.lt 0 (f x)
      ⊢ Exists fun f => Ne (f x) 0
    -/
    exact ⟨f, hf.ne'⟩⟩
    /-
      🎉 no goals
    -/


instance {E 𝕜 : Type*} [RCLike 𝕜] [NormedAddCommGroup E] [NormedSpace 𝕜 E] : SeparatingDual 𝕜 E :=
  ⟨fun x hx ↦ by
    /-
      E : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      hx : Ne x 0
      ⊢ Exists fun f => Ne (f x) 0
    -/
    rcases exists_dual_vector 𝕜 x hx with ⟨f, -, hf⟩
    /-
      case intro.intro
      E : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      hx : Ne x 0
      f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hf : Eq (f x) ↑(Norm.norm x)
      ⊢ Exists fun f => Ne (f x) 0
    -/
    refine ⟨f, ?_⟩
    /-
      case intro.intro
      E : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      hx : Ne x 0
      f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hf : Eq (f x) ↑(Norm.norm x)
      ⊢ Ne (f x) 0
    -/
    simpa [hf] using hx⟩
    /-
      🎉 no goals
    -/


lemma exists_ne_zero {x : V} (hx : x ≠ 0) :
    ∃ f : V →L[R] R, f x ≠ 0 :=
  exists_ne_zero' x hx


theorem exists_separating_of_ne {x y : V} (h : x ≠ y) :
    ∃ f : V →L[R] R, f x ≠ f y := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup V
    inst✝³ : TopologicalSpace V
    inst✝² : TopologicalSpace R
    inst✝¹ : Module R V
    inst✝ : SeparatingDual R V
    x y : V
    h : Ne x y
    ⊢ Exists fun f => Ne (f x) (f y)
  -/
  rcases exists_ne_zero (R := R) (sub_ne_zero_of_ne h) with ⟨f, hf⟩
  /-
    case intro
    R : Type u_1
    V : Type u_2
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup V
    inst✝³ : TopologicalSpace V
    inst✝² : TopologicalSpace R
    inst✝¹ : Module R V
    inst✝ : SeparatingDual R V
    x y : V
    h : Ne x y
    f : ContinuousLinearMap (RingHom.id R) V R
    hf : Ne (f (HSub.hSub x y)) 0
    ⊢ Exists fun f => Ne (f x) (f y)
  -/
  exact ⟨f, by simpa [sub_ne_zero] using hf⟩
  /-
    🎉 no goals
  -/


protected theorem t1Space [T1Space R] : T1Space V := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace V
    inst✝³ : TopologicalSpace R
    inst✝² : Module R V
    inst✝¹ : SeparatingDual R V
    inst✝ : T1Space R
    ⊢ T1Space V
  -/
  apply t1Space_iff_exists_open.2 (fun x y hxy ↦ ?_)
  /-
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace V
    inst✝³ : TopologicalSpace R
    inst✝² : Module R V
    inst✝¹ : SeparatingDual R V
    inst✝ : T1Space R
    x y : V
    hxy : Ne x y
    ⊢ Exists fun U => And (IsOpen U) (And (Membership.mem U x) (Not (Membership.me …
  -/
  rcases exists_separating_of_ne (R := R) hxy with ⟨f, hf⟩
  /-
    case intro
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace V
    inst✝³ : TopologicalSpace R
    inst✝² : Module R V
    inst✝¹ : SeparatingDual R V
    inst✝ : T1Space R
    x y : V
    hxy : Ne x y
    f : ContinuousLinearMap (RingHom.id R) V R
    hf : Ne (f x) (f y)
    ⊢ Exists fun U => And (IsOpen U) (And (Membership.mem U x) (Not (Membership.me …
  -/
  exact ⟨f ⁻¹' {f y}ᶜ, isOpen_compl_singleton.preimage f.continuous, hf, by simp⟩
  /-
    🎉 no goals
  -/


protected theorem t2Space [T2Space R] : T2Space V := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace V
    inst✝³ : TopologicalSpace R
    inst✝² : Module R V
    inst✝¹ : SeparatingDual R V
    inst✝ : T2Space R
    ⊢ T2Space V
  -/
  apply (t2Space_iff _).2 (fun {x} {y} hxy ↦ ?_)
  /-
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace V
    inst✝³ : TopologicalSpace R
    inst✝² : Module R V
    inst✝¹ : SeparatingDual R V
    inst✝ : T2Space R
    x y : V
    hxy : Ne x y
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Members …
  -/
  rcases exists_separating_of_ne (R := R) hxy with ⟨f, hf⟩
  /-
    case intro
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace V
    inst✝³ : TopologicalSpace R
    inst✝² : Module R V
    inst✝¹ : SeparatingDual R V
    inst✝ : T2Space R
    x y : V
    hxy : Ne x y
    f : ContinuousLinearMap (RingHom.id R) V R
    hf : Ne (f x) (f y)
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Members …
  -/
  exact separated_by_continuous f.continuous hf
  /-
    🎉 no goals
  -/


theorem _root_.separatingDual_iff_injective : SeparatingDual R V ↔
    Function.Injective (ContinuousLinearMap.coeLM (R := R) R (M := V) (N₃ := R)).flip := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝⁵ : Field R
    inst✝⁴ : AddCommGroup V
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSpace V
    inst✝¹ : TopologicalRing R
    inst✝ : Module R V
    ⊢ Iff (SeparatingDual R V) (Function.Injective ⇑(ContinuousLinearMap.coeLM R). …
  -/
  simp_rw [separatingDual_def, Ne, injective_iff_map_eq_zero]
  /-
    R : Type u_1
    V : Type u_2
    inst✝⁵ : Field R
    inst✝⁴ : AddCommGroup V
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSpace V
    inst✝¹ : TopologicalRing R
    inst✝ : Module R V
    ⊢ Iff (∀ (x : V), Not (Eq x 0) → Exists fun f => Not (Eq (f x) 0)) (∀ (a : V), …
  -/
  congrm ∀ v, ?_
  /-
    case a
    R : Type u_1
    V : Type u_2
    inst✝⁵ : Field R
    inst✝⁴ : AddCommGroup V
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSpace V
    inst✝¹ : TopologicalRing R
    inst✝ : Module R V
    v : V
    ⊢ Iff (Not (Eq v 0) → Exists fun f => Not (Eq (f v) 0)) (Eq ((ContinuousLinear …
  -/
  rw [not_imp_comm, LinearMap.ext_iff]
  /-
    case a
    R : Type u_1
    V : Type u_2
    inst✝⁵ : Field R
    inst✝⁴ : AddCommGroup V
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSpace V
    inst✝¹ : TopologicalRing R
    inst✝ : Module R V
    v : V
    ⊢ Iff (Not (Exists fun f => Not (Eq (f v) 0)) → Eq v 0) ((∀ (x : ContinuousLin …
  -/
  push_neg; rfl
            /-
              🎉 no goals
            -/


open Function in
/-- Given a finite-dimensional subspace `W` of a space `V` with separating dual, any
  linear functional on `W` extends to a continuous linear functional on `V`.
  This is stated more generally for an injective linear map from `W` to `V`. -/
theorem dualMap_surjective_iff {W} [AddCommGroup W] [Module R W] [FiniteDimensional R W]
    {f : W →ₗ[R] V} : Surjective (f.dualMap ∘ ContinuousLinearMap.toLinearMap) ↔ Injective f := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝⁹ : Field R
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace V
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : Module R V
    inst✝³ : SeparatingDual R V
    W : Type u_3
    inst✝² : AddCommGroup W
    inst✝¹ : Module R W
    inst✝ : FiniteDimensional R W
    f : LinearMap (RingHom.id R) W V
    ⊢ Iff (Function.Surjective (Function.comp (⇑f.dualMap) ContinuousLinearMap.toL …
  -/
  constructor <;> intro hf
    /-
      case mp
      R : Type u_1
      V : Type u_2
      inst✝⁹ : Field R
      inst✝⁸ : AddCommGroup V
      inst✝⁷ : TopologicalSpace R
      inst✝⁶ : TopologicalSpace V
      inst✝⁵ : TopologicalRing R
      inst✝⁴ : Module R V
      inst✝³ : SeparatingDual R V
      W : Type u_3
      inst✝² : AddCommGroup W
      inst✝¹ : Module R W
      inst✝ : FiniteDimensional R W
      f : LinearMap (RingHom.id R) W V
      hf : Function.Surjective (Function.comp (⇑f.dualMap) ContinuousLinearMap.toLin …
      ⊢ Function.Injective ⇑f
    -/
  · exact LinearMap.dualMap_surjective_iff.mp hf.of_comp
    /-
      🎉 no goals
    -/
  /-
    case mpr
    R : Type u_1
    V : Type u_2
    inst✝⁹ : Field R
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace V
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : Module R V
    inst✝³ : SeparatingDual R V
    W : Type u_3
    inst✝² : AddCommGroup W
    inst✝¹ : Module R W
    inst✝ : FiniteDimensional R W
    f : LinearMap (RingHom.id R) W V
    hf : Function.Injective ⇑f
    ⊢ Function.Surjective (Function.comp (⇑f.dualMap) ContinuousLinearMap.toLinear …
  -/
  have := (separatingDual_iff_injective.mp ‹_›).comp hf
  /-
    case mpr
    R : Type u_1
    V : Type u_2
    inst✝⁹ : Field R
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace V
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : Module R V
    inst✝³ : SeparatingDual R V
    W : Type u_3
    inst✝² : AddCommGroup W
    inst✝¹ : Module R W
    inst✝ : FiniteDimensional R W
    f : LinearMap (RingHom.id R) W V
    hf : Function.Injective ⇑f
    this : Function.Injective (Function.comp ⇑(ContinuousLinearMap.coeLM R).flip ⇑f)
    ⊢ Function.Surjective (Function.comp (⇑f.dualMap) ContinuousLinearMap.toLinear …
  -/
  rw [← LinearMap.coe_comp] at this
  /-
    case mpr
    R : Type u_1
    V : Type u_2
    inst✝⁹ : Field R
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace V
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : Module R V
    inst✝³ : SeparatingDual R V
    W : Type u_3
    inst✝² : AddCommGroup W
    inst✝¹ : Module R W
    inst✝ : FiniteDimensional R W
    f : LinearMap (RingHom.id R) W V
    hf : Function.Injective ⇑f
    this : Function.Injective ⇑((ContinuousLinearMap.coeLM R).flip.comp f)
    ⊢ Function.Surjective (Function.comp (⇑f.dualMap) ContinuousLinearMap.toLinear …
  -/
  exact LinearMap.flip_surjective_iff₁.mpr this
  /-
    🎉 no goals
  -/


lemma exists_eq_one {x : V} (hx : x ≠ 0) :
    ∃ f : V →L[R] R, f x = 1 := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Field R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace R
    inst✝³ : TopologicalSpace V
    inst✝² : TopologicalRing R
    inst✝¹ : Module R V
    inst✝ : SeparatingDual R V
    x : V
    hx : Ne x 0
    ⊢ Exists fun f => Eq (f x) 1
  -/
  rcases exists_ne_zero (R := R) hx with ⟨f, hf⟩
  /-
    case intro
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Field R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace R
    inst✝³ : TopologicalSpace V
    inst✝² : TopologicalRing R
    inst✝¹ : Module R V
    inst✝ : SeparatingDual R V
    x : V
    hx : Ne x 0
    f : ContinuousLinearMap (RingHom.id R) V R
    hf : Ne (f x) 0
    ⊢ Exists fun f => Eq (f x) 1
  -/
  exact ⟨(f x)⁻¹ • f, inv_mul_cancel₀ hf⟩
  /-
    🎉 no goals
  -/


theorem exists_eq_one_ne_zero_of_ne_zero_pair {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    ∃ f : V →L[R] R, f x = 1 ∧ f y ≠ 0 := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Field R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace R
    inst✝³ : TopologicalSpace V
    inst✝² : TopologicalRing R
    inst✝¹ : Module R V
    inst✝ : SeparatingDual R V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Exists fun f => And (Eq (f x) 1) (Ne (f y) 0)
  -/
  obtain ⟨u, ux⟩ : ∃ u : V →L[R] R, u x = 1 := exists_eq_one hx
  /-
    case intro
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Field R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace R
    inst✝³ : TopologicalSpace V
    inst✝² : TopologicalRing R
    inst✝¹ : Module R V
    inst✝ : SeparatingDual R V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    u : ContinuousLinearMap (RingHom.id R) V R
    ux : Eq (u x) 1
    ⊢ Exists fun f => And (Eq (f x) 1) (Ne (f y) 0)
  -/
  rcases ne_or_eq (u y) 0 with uy|uy
    /-
      case intro.inl
      R : Type u_1
      V : Type u_2
      inst✝⁶ : Field R
      inst✝⁵ : AddCommGroup V
      inst✝⁴ : TopologicalSpace R
      inst✝³ : TopologicalSpace V
      inst✝² : TopologicalRing R
      inst✝¹ : Module R V
      inst✝ : SeparatingDual R V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      u : ContinuousLinearMap (RingHom.id R) V R
      ux : Eq (u x) 1
      uy : Ne (u y) 0
      ⊢ Exists fun f => And (Eq (f x) 1) (Ne (f y) 0)
    -/
  · exact ⟨u, ux, uy⟩
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Field R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace R
    inst✝³ : TopologicalSpace V
    inst✝² : TopologicalRing R
    inst✝¹ : Module R V
    inst✝ : SeparatingDual R V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    u : ContinuousLinearMap (RingHom.id R) V R
    ux : Eq (u x) 1
    uy : Eq (u y) 0
    ⊢ Exists fun f => And (Eq (f x) 1) (Ne (f y) 0)
  -/
  obtain ⟨v, vy⟩ : ∃ v : V →L[R] R, v y = 1 := exists_eq_one hy
  /-
    case intro.inr.intro
    R : Type u_1
    V : Type u_2
    inst✝⁶ : Field R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : TopologicalSpace R
    inst✝³ : TopologicalSpace V
    inst✝² : TopologicalRing R
    inst✝¹ : Module R V
    inst✝ : SeparatingDual R V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    u : ContinuousLinearMap (RingHom.id R) V R
    ux : Eq (u x) 1
    uy : Eq (u y) 0
    v : ContinuousLinearMap (RingHom.id R) V R
    vy : Eq (v y) 1
    ⊢ Exists fun f => And (Eq (f x) 1) (Ne (f y) 0)
  -/
  rcases ne_or_eq (v x) 0 with vx|vx
    /-
      case intro.inr.intro.inl
      R : Type u_1
      V : Type u_2
      inst✝⁶ : Field R
      inst✝⁵ : AddCommGroup V
      inst✝⁴ : TopologicalSpace R
      inst✝³ : TopologicalSpace V
      inst✝² : TopologicalRing R
      inst✝¹ : Module R V
      inst✝ : SeparatingDual R V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      u : ContinuousLinearMap (RingHom.id R) V R
      ux : Eq (u x) 1
      uy : Eq (u y) 0
      v : ContinuousLinearMap (RingHom.id R) V R
      vy : Eq (v y) 1
      vx : Ne (v x) 0
      ⊢ Exists fun f => And (Eq (f x) 1) (Ne (f y) 0)
    -/
  · exact ⟨(v x)⁻¹ • v, inv_mul_cancel₀ vx, show (v x)⁻¹ * v y ≠ 0 by simp [vx, vy]⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.intro.inr
      R : Type u_1
      V : Type u_2
      inst✝⁶ : Field R
      inst✝⁵ : AddCommGroup V
      inst✝⁴ : TopologicalSpace R
      inst✝³ : TopologicalSpace V
      inst✝² : TopologicalRing R
      inst✝¹ : Module R V
      inst✝ : SeparatingDual R V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      u : ContinuousLinearMap (RingHom.id R) V R
      ux : Eq (u x) 1
      uy : Eq (u y) 0
      v : ContinuousLinearMap (RingHom.id R) V R
      vy : Eq (v y) 1
      vx : Eq (v x) 0
      ⊢ Exists fun f => And (Eq (f x) 1) (Ne (f y) 0)
    -/
  · exact ⟨u + v, by simp [ux, vx], by simp [uy, vy]⟩
    /-
      🎉 no goals
    -/


/-- In a topological vector space with separating dual, the group of continuous linear equivalences
acts transitively on the set of nonzero vectors: given two nonzero vectors `x` and `y`, there
exists `A : V ≃L[R] V` mapping `x` to `y`. -/
theorem exists_continuousLinearEquiv_apply_eq [ContinuousSMul R V]
    {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    ∃ A : V ≃L[R] V, A x = y := by
  obtain ⟨G, Gx, Gy⟩ : ∃ G : V →L[R] R, G x = 1 ∧ G y ≠ 0 :=
    exists_eq_one_ne_zero_of_ne_zero_pair hx hy
  let A : V ≃L[R] V :=
  { toFun := fun z ↦ z + G z • (y - x)
    invFun := fun z ↦ z + ((G y) ⁻¹ * G z) • (x - y)
    map_add' := fun a b ↦ by simp [add_smul]; abel
    map_smul' := by simp [smul_smul]
    left_inv := fun z ↦ by
      simp only [id_eq, eq_mpr_eq_cast, RingHom.id_apply, smul_eq_mul, AddHom.toFun_eq_coe,
        -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to change `map_smulₛₗ` into `map_smulₛₗ _`
        AddHom.coe_mk, map_add, map_smulₛₗ _, map_sub, Gx, mul_sub, mul_one, add_sub_cancel]
      rw [mul_comm (G z), ← mul_assoc, inv_mul_cancel₀ Gy]
      simp only [smul_sub, one_mul]
      abel
    right_inv := fun z ↦ by
        -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to change `map_smulₛₗ` into `map_smulₛₗ _`
      simp only [map_add, map_smulₛₗ _, map_mul, map_inv₀, RingHom.id_apply, map_sub, Gx,
        smul_eq_mul, mul_sub, mul_one]
      rw [mul_comm _ (G y), ← mul_assoc, mul_inv_cancel₀ Gy]
      simp only [smul_sub, one_mul, add_sub_cancel]
      abel
    continuous_toFun := continuous_id.add (G.continuous.smul continuous_const)
    continuous_invFun :=
      continuous_id.add ((continuous_const.mul G.continuous).smul continuous_const) }
  /-
    case intro.intro
    R : Type u_1
    V : Type u_2
    inst✝⁸ : Field R
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : TopologicalSpace R
    inst✝⁵ : TopologicalSpace V
    inst✝⁴ : TopologicalRing R
    inst✝³ : Module R V
    inst✝² : SeparatingDual R V
    inst✝¹ : TopologicalAddGroup V
    inst✝ : ContinuousSMul R V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    G : ContinuousLinearMap (RingHom.id R) V R
    Gx : Eq (G x) 1
    Gy : Ne (G y) 0
    A : ContinuousLinearEquiv (RingHom.id R) V V := { toFun := fun z => HAdd.hAdd  …
    ⊢ Exists fun A => Eq (A x) y
  -/
  exact ⟨A, show x + G x • (y - x) = y by simp [Gx]⟩
  /-
    🎉 no goals
  -/


/-- If a space of linear maps from `E` to `F` is complete, and `E` is nontrivial, then `F` is
complete. -/
lemma completeSpace_of_completeSpace_continuousLinearMap [CompleteSpace (E →L[𝕜] F)] :
    CompleteSpace F := by
  /-
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : SeparatingDual 𝕜 E
    inst✝¹ : Nontrivial E
    inst✝ : CompleteSpace (ContinuousLinearMap (RingHom.id 𝕜) E F)
    ⊢ CompleteSpace F
  -/
  refine Metric.complete_of_cauchySeq_tendsto fun f hf => ?_
  /-
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : SeparatingDual 𝕜 E
    inst✝¹ : Nontrivial E
    inst✝ : CompleteSpace (ContinuousLinearMap (RingHom.id 𝕜) E F)
    f : Nat → F
    hf : CauchySeq f
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  obtain ⟨v, hv⟩ : ∃ (v : E), v ≠ 0 := exists_ne 0
  /-
    case intro
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : SeparatingDual 𝕜 E
    inst✝¹ : Nontrivial E
    inst✝ : CompleteSpace (ContinuousLinearMap (RingHom.id 𝕜) E F)
    f : Nat → F
    hf : CauchySeq f
    v : E
    hv : Ne v 0
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  obtain ⟨φ, hφ⟩ : ∃ φ : E →L[𝕜] 𝕜, φ v = 1 := exists_eq_one hv
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : SeparatingDual 𝕜 E
    inst✝¹ : Nontrivial E
    inst✝ : CompleteSpace (ContinuousLinearMap (RingHom.id 𝕜) E F)
    f : Nat → F
    hf : CauchySeq f
    v : E
    hv : Ne v 0
    φ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hφ : Eq (φ v) 1
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  let g : ℕ → (E →L[𝕜] F) := fun n ↦ ContinuousLinearMap.smulRightL 𝕜 E F φ (f n)
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : SeparatingDual 𝕜 E
    inst✝¹ : Nontrivial E
    inst✝ : CompleteSpace (ContinuousLinearMap (RingHom.id 𝕜) E F)
    f : Nat → F
    hf : CauchySeq f
    v : E
    hv : Ne v 0
    φ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hφ : Eq (φ v) 1
    g : Nat → ContinuousLinearMap (RingHom.id 𝕜) E F := fun n => ((ContinuousLinea …
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  have : CauchySeq g := (ContinuousLinearMap.smulRightL 𝕜 E F φ).lipschitz.cauchySeq_comp hf
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : SeparatingDual 𝕜 E
    inst✝¹ : Nontrivial E
    inst✝ : CompleteSpace (ContinuousLinearMap (RingHom.id 𝕜) E F)
    f : Nat → F
    hf : CauchySeq f
    v : E
    hv : Ne v 0
    φ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hφ : Eq (φ v) 1
    g : Nat → ContinuousLinearMap (RingHom.id 𝕜) E F := fun n => ((ContinuousLinea …
    this : CauchySeq g
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  obtain ⟨a, ha⟩ : ∃ a, Tendsto g atTop (𝓝 a) := cauchy_iff_exists_le_nhds.mp this
  /-
    case intro.intro.intro
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : SeparatingDual 𝕜 E
    inst✝¹ : Nontrivial E
    inst✝ : CompleteSpace (ContinuousLinearMap (RingHom.id 𝕜) E F)
    f : Nat → F
    hf : CauchySeq f
    v : E
    hv : Ne v 0
    φ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hφ : Eq (φ v) 1
    g : Nat → ContinuousLinearMap (RingHom.id 𝕜) E F := fun n => ((ContinuousLinea …
    this : CauchySeq g
    a : ContinuousLinearMap (RingHom.id 𝕜) E F
    ha : Filter.Tendsto g Filter.atTop (nhds a)
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  refine ⟨a v, ?_⟩
  have : Tendsto (fun n ↦ g n v) atTop (𝓝 (a v)) := by
    have : Continuous (fun (i : E →L[𝕜] F) ↦ i v) := by fun_prop
    exact (this.tendsto _).comp ha
  /-
    case intro.intro.intro
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : SeparatingDual 𝕜 E
    inst✝¹ : Nontrivial E
    inst✝ : CompleteSpace (ContinuousLinearMap (RingHom.id 𝕜) E F)
    f : Nat → F
    hf : CauchySeq f
    v : E
    hv : Ne v 0
    φ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hφ : Eq (φ v) 1
    g : Nat → ContinuousLinearMap (RingHom.id 𝕜) E F := fun n => ((ContinuousLinea …
    this✝ : CauchySeq g
    a : ContinuousLinearMap (RingHom.id 𝕜) E F
    ha : Filter.Tendsto g Filter.atTop (nhds a)
    this : Filter.Tendsto (fun n => (g n) v) Filter.atTop (nhds (a v))
    ⊢ Filter.Tendsto f Filter.atTop (nhds (a v))
  -/
  simpa [g, ContinuousLinearMap.smulRightL, hφ]
  /-
    🎉 no goals
  -/


lemma completeSpace_continuousLinearMap_iff :
    CompleteSpace (E →L[𝕜] F) ↔ CompleteSpace F :=
  ⟨fun _h ↦ completeSpace_of_completeSpace_continuousLinearMap 𝕜 E F, fun _h ↦ inferInstance⟩


/-- If a space of multilinear maps from `Π i, E i` to `F` is complete, and each `E i` has a nonzero
element, then `F` is complete. -/
lemma completeSpace_of_completeSpace_continuousMultilinearMap
    [CompleteSpace (ContinuousMultilinearMap 𝕜 M F)]
    {m : ∀ i, M i} (hm : ∀ i, m i ≠ 0) : CompleteSpace F := by
  /-
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    ι : Type u_6
    inst✝⁴ : Finite ι
    M : ι → Type u_7
    inst✝³ : (i : ι) → NormedAddCommGroup (M i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (M i)
    inst✝¹ : ∀ (i : ι), SeparatingDual 𝕜 (M i)
    inst✝ : CompleteSpace (ContinuousMultilinearMap 𝕜 M F)
    m : (i : ι) → M i
    hm : ∀ (i : ι), Ne (m i) 0
    ⊢ CompleteSpace F
  -/
  refine Metric.complete_of_cauchySeq_tendsto fun f hf => ?_
  /-
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    ι : Type u_6
    inst✝⁴ : Finite ι
    M : ι → Type u_7
    inst✝³ : (i : ι) → NormedAddCommGroup (M i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (M i)
    inst✝¹ : ∀ (i : ι), SeparatingDual 𝕜 (M i)
    inst✝ : CompleteSpace (ContinuousMultilinearMap 𝕜 M F)
    m : (i : ι) → M i
    hm : ∀ (i : ι), Ne (m i) 0
    f : Nat → F
    hf : CauchySeq f
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  have : ∀ i, ∃ φ : M i →L[𝕜] 𝕜, φ (m i) = 1 := fun i ↦ exists_eq_one (hm i)
  /-
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    ι : Type u_6
    inst✝⁴ : Finite ι
    M : ι → Type u_7
    inst✝³ : (i : ι) → NormedAddCommGroup (M i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (M i)
    inst✝¹ : ∀ (i : ι), SeparatingDual 𝕜 (M i)
    inst✝ : CompleteSpace (ContinuousMultilinearMap 𝕜 M F)
    m : (i : ι) → M i
    hm : ∀ (i : ι), Ne (m i) 0
    f : Nat → F
    hf : CauchySeq f
    this : ∀ (i : ι), Exists fun φ => Eq (φ (m i)) 1
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  choose φ hφ using this
  /-
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    ι : Type u_6
    inst✝⁴ : Finite ι
    M : ι → Type u_7
    inst✝³ : (i : ι) → NormedAddCommGroup (M i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (M i)
    inst✝¹ : ∀ (i : ι), SeparatingDual 𝕜 (M i)
    inst✝ : CompleteSpace (ContinuousMultilinearMap 𝕜 M F)
    m : (i : ι) → M i
    hm : ∀ (i : ι), Ne (m i) 0
    f : Nat → F
    hf : CauchySeq f
    φ : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (M i) 𝕜
    hφ : ∀ (i : ι), Eq ((φ i) (m i)) 1
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  cases nonempty_fintype ι
  let g : ℕ → (ContinuousMultilinearMap 𝕜 M F) := fun n ↦
    compContinuousLinearMapL φ
    (ContinuousMultilinearMap.smulRightL 𝕜 _ F ((ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι 𝕜)) (f n))
  have : CauchySeq g := by
    refine (ContinuousLinearMap.lipschitz _).cauchySeq_comp ?_
    exact (ContinuousLinearMap.lipschitz _).cauchySeq_comp hf
  /-
    case intro
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    ι : Type u_6
    inst✝⁴ : Finite ι
    M : ι → Type u_7
    inst✝³ : (i : ι) → NormedAddCommGroup (M i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (M i)
    inst✝¹ : ∀ (i : ι), SeparatingDual 𝕜 (M i)
    inst✝ : CompleteSpace (ContinuousMultilinearMap 𝕜 M F)
    m : (i : ι) → M i
    hm : ∀ (i : ι), Ne (m i) 0
    f : Nat → F
    hf : CauchySeq f
    φ : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (M i) 𝕜
    hφ : ∀ (i : ι), Eq ((φ i) (m i)) 1
    val✝ : Fintype ι
    g : Nat → ContinuousMultilinearMap 𝕜 M F := fun n => (ContinuousMultilinearMap …
    this : CauchySeq g
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  obtain ⟨a, ha⟩ : ∃ a, Tendsto g atTop (𝓝 a) := cauchy_iff_exists_le_nhds.mp this
  /-
    case intro.intro
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    ι : Type u_6
    inst✝⁴ : Finite ι
    M : ι → Type u_7
    inst✝³ : (i : ι) → NormedAddCommGroup (M i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (M i)
    inst✝¹ : ∀ (i : ι), SeparatingDual 𝕜 (M i)
    inst✝ : CompleteSpace (ContinuousMultilinearMap 𝕜 M F)
    m : (i : ι) → M i
    hm : ∀ (i : ι), Ne (m i) 0
    f : Nat → F
    hf : CauchySeq f
    φ : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (M i) 𝕜
    hφ : ∀ (i : ι), Eq ((φ i) (m i)) 1
    val✝ : Fintype ι
    g : Nat → ContinuousMultilinearMap 𝕜 M F := fun n => (ContinuousMultilinearMap …
    this : CauchySeq g
    a : ContinuousMultilinearMap 𝕜 M F
    ha : Filter.Tendsto g Filter.atTop (nhds a)
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  refine ⟨a m, ?_⟩
  /-
    case intro.intro
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    ι : Type u_6
    inst✝⁴ : Finite ι
    M : ι → Type u_7
    inst✝³ : (i : ι) → NormedAddCommGroup (M i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (M i)
    inst✝¹ : ∀ (i : ι), SeparatingDual 𝕜 (M i)
    inst✝ : CompleteSpace (ContinuousMultilinearMap 𝕜 M F)
    m : (i : ι) → M i
    hm : ∀ (i : ι), Ne (m i) 0
    f : Nat → F
    hf : CauchySeq f
    φ : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (M i) 𝕜
    hφ : ∀ (i : ι), Eq ((φ i) (m i)) 1
    val✝ : Fintype ι
    g : Nat → ContinuousMultilinearMap 𝕜 M F := fun n => (ContinuousMultilinearMap …
    this : CauchySeq g
    a : ContinuousMultilinearMap 𝕜 M F
    ha : Filter.Tendsto g Filter.atTop (nhds a)
    ⊢ Filter.Tendsto f Filter.atTop (nhds (a m))
  -/
  have : Tendsto (fun n ↦ g n m) atTop (𝓝 (a m)) := ((continuous_eval_const _).tendsto _).comp ha
  /-
    case intro.intro
    𝕜 : Type u_3
    F : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    ι : Type u_6
    inst✝⁴ : Finite ι
    M : ι → Type u_7
    inst✝³ : (i : ι) → NormedAddCommGroup (M i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (M i)
    inst✝¹ : ∀ (i : ι), SeparatingDual 𝕜 (M i)
    inst✝ : CompleteSpace (ContinuousMultilinearMap 𝕜 M F)
    m : (i : ι) → M i
    hm : ∀ (i : ι), Ne (m i) 0
    f : Nat → F
    hf : CauchySeq f
    φ : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (M i) 𝕜
    hφ : ∀ (i : ι), Eq ((φ i) (m i)) 1
    val✝ : Fintype ι
    g : Nat → ContinuousMultilinearMap 𝕜 M F := fun n => (ContinuousMultilinearMap …
    this✝ : CauchySeq g
    a : ContinuousMultilinearMap 𝕜 M F
    ha : Filter.Tendsto g Filter.atTop (nhds a)
    this : Filter.Tendsto (fun n => (g n) m) Filter.atTop (nhds (a m))
    ⊢ Filter.Tendsto f Filter.atTop (nhds (a m))
  -/
  simpa [g, hφ]
  /-
    🎉 no goals
  -/


lemma completeSpace_continuousMultilinearMap_iff {m : ∀ i, M i} (hm : ∀ i, m i ≠ 0) :
    CompleteSpace (ContinuousMultilinearMap 𝕜 M F) ↔ CompleteSpace F :=
  ⟨fun _h ↦ completeSpace_of_completeSpace_continuousMultilinearMap 𝕜 F hm, fun _h ↦ inferInstance⟩


