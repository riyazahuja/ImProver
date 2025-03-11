open Cardinal in
theorem MvRatFunc.rank_eq_max_lift
    {σ : Type u} {F : Type v} [Field F] [Nonempty σ] :
    Module.rank F (FractionRing (MvPolynomial σ F)) = lift.{u} #F ⊔ lift.{v} #σ ⊔ ℵ₀ := by
  /-
    σ : Type u
    F : Type v
    inst✝¹ : Field F
    inst✝ : Nonempty σ
    ⊢ Eq (Module.rank F (FractionRing (MvPolynomial σ F))) (Max.max (Max.max (Card …
  -/
  let R := MvPolynomial σ F
  /-
    σ : Type u
    F : Type v
    inst✝¹ : Field F
    inst✝ : Nonempty σ
    R : Type (max u v) := MvPolynomial σ F
    ⊢ Eq (Module.rank F (FractionRing (MvPolynomial σ F))) (Max.max (Max.max (Card …
  -/
  let K := FractionRing R
  /-
    σ : Type u
    F : Type v
    inst✝¹ : Field F
    inst✝ : Nonempty σ
    R : Type (max u v) := MvPolynomial σ F
    K : Type (max u v) := FractionRing R
    ⊢ Eq (Module.rank F (FractionRing (MvPolynomial σ F))) (Max.max (Max.max (Card …
  -/
  refine ((rank_le_card _ _).trans ?_).antisymm ?_
    /-
      case refine_1
      σ : Type u
      F : Type v
      inst✝¹ : Field F
      inst✝ : Nonempty σ
      R : Type (max u v) := MvPolynomial σ F
      K : Type (max u v) := FractionRing R
      ⊢ LE.le (Cardinal.mk (FractionRing (MvPolynomial σ F))) (Max.max (Max.max (Car …
    -/
  · rw [FractionRing.cardinalMk, MvPolynomial.cardinalMk_eq_max_lift]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    σ : Type u
    F : Type v
    inst✝¹ : Field F
    inst✝ : Nonempty σ
    R : Type (max u v) := MvPolynomial σ F
    K : Type (max u v) := FractionRing R
    ⊢ LE.le (Max.max (Max.max (Cardinal.lift.{u, v} (Cardinal.mk F)) (Cardinal.lif …
  -/
  have hinj := IsFractionRing.injective R K
  /-
    case refine_2
    σ : Type u
    F : Type v
    inst✝¹ : Field F
    inst✝ : Nonempty σ
    R : Type (max u v) := MvPolynomial σ F
    K : Type (max u v) := FractionRing R
    hinj : Function.Injective ⇑(algebraMap R K)
    ⊢ LE.le (Max.max (Max.max (Cardinal.lift.{u, v} (Cardinal.mk F)) (Cardinal.lif …
  -/
  have h1 := (IsScalarTower.toAlgHom F R K).toLinearMap.rank_le_of_injective hinj
  /-
    case refine_2
    σ : Type u
    F : Type v
    inst✝¹ : Field F
    inst✝ : Nonempty σ
    R : Type (max u v) := MvPolynomial σ F
    K : Type (max u v) := FractionRing R
    hinj : Function.Injective ⇑(algebraMap R K)
    h1 : LE.le (Module.rank F R) (Module.rank F K)
    ⊢ LE.le (Max.max (Max.max (Cardinal.lift.{u, v} (Cardinal.mk F)) (Cardinal.lif …
  -/
  rw [MvPolynomial.rank_eq_lift, mk_finsupp_nat, lift_max, lift_aleph0, max_le_iff] at h1
  /-
    case refine_2
    σ : Type u
    F : Type v
    inst✝¹ : Field F
    inst✝ : Nonempty σ
    R : Type (max u v) := MvPolynomial σ F
    K : Type (max u v) := FractionRing R
    hinj : Function.Injective ⇑(algebraMap R K)
    h1 : And (LE.le (Cardinal.lift.{v, u} (Cardinal.mk σ)) (Module.rank F K)) (LE. …
    ⊢ LE.le (Max.max (Max.max (Cardinal.lift.{u, v} (Cardinal.mk F)) (Cardinal.lif …
  -/
  obtain ⟨i⟩ := ‹Nonempty σ›
  have hx : Transcendental F (algebraMap R K (MvPolynomial.X i)) :=
    (transcendental_algebraMap_iff hinj).2 (MvPolynomial.transcendental_X F i)
  /-
    case refine_2.intro
    σ : Type u
    F : Type v
    inst✝¹ : Field F
    inst✝ : Nonempty σ
    R : Type (max u v) := MvPolynomial σ F
    K : Type (max u v) := FractionRing R
    hinj : Function.Injective ⇑(algebraMap R K)
    h1 : And (LE.le (Cardinal.lift.{v, u} (Cardinal.mk σ)) (Module.rank F K)) (LE. …
    i : σ
    hx : Transcendental F ((algebraMap R K) (MvPolynomial.X i))
    ⊢ LE.le (Max.max (Max.max (Cardinal.lift.{u, v} (Cardinal.mk F)) (Cardinal.lif …
  -/
  have h2 := hx.linearIndependent_sub_inv.cardinal_lift_le_rank
  /-
    case refine_2.intro
    σ : Type u
    F : Type v
    inst✝¹ : Field F
    inst✝ : Nonempty σ
    R : Type (max u v) := MvPolynomial σ F
    K : Type (max u v) := FractionRing R
    hinj : Function.Injective ⇑(algebraMap R K)
    h1 : And (LE.le (Cardinal.lift.{v, u} (Cardinal.mk σ)) (Module.rank F K)) (LE. …
    i : σ
    hx : Transcendental F ((algebraMap R K) (MvPolynomial.X i))
    h2 : LE.le (Cardinal.lift.{max u v, v} (Cardinal.mk F)) (Cardinal.lift.{v, max …
    ⊢ LE.le (Max.max (Max.max (Cardinal.lift.{u, v} (Cardinal.mk F)) (Cardinal.lif …
  -/
  rw [lift_id'.{v, u}, lift_umax.{v, u}] at h2
  /-
    case refine_2.intro
    σ : Type u
    F : Type v
    inst✝¹ : Field F
    inst✝ : Nonempty σ
    R : Type (max u v) := MvPolynomial σ F
    K : Type (max u v) := FractionRing R
    hinj : Function.Injective ⇑(algebraMap R K)
    h1 : And (LE.le (Cardinal.lift.{v, u} (Cardinal.mk σ)) (Module.rank F K)) (LE. …
    i : σ
    hx : Transcendental F ((algebraMap R K) (MvPolynomial.X i))
    h2 : LE.le (Cardinal.lift.{u, v} (Cardinal.mk F)) (Module.rank F K)
    ⊢ LE.le (Max.max (Max.max (Cardinal.lift.{u, v} (Cardinal.mk F)) (Cardinal.lif …
  -/
  exact max_le (max_le h2 h1.1) h1.2
  /-
    🎉 no goals
  -/

