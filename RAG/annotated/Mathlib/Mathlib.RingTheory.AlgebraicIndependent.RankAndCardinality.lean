theorem IsTranscendenceBasis.lift_cardinalMk_eq_max_lift
    {F : Type u} {E : Type v} [CommRing F] [Nontrivial F] [CommRing E] [IsDomain E] [Algebra F E]
    {ι : Type w} {x : ι → E} [Nonempty ι] (hx : IsTranscendenceBasis F x) :
    lift.{max u w} #E = lift.{max v w} #F ⊔ lift.{max u v} #ι ⊔ ℵ₀ := by
  /-
    F : Type u
    E : Type v
    inst✝⁵ : CommRing F
    inst✝⁴ : Nontrivial F
    inst✝³ : CommRing E
    inst✝² : IsDomain E
    inst✝¹ : Algebra F E
    ι : Type w
    x : ι → E
    inst✝ : Nonempty ι
    hx : IsTranscendenceBasis F x
    ⊢ Eq (Cardinal.lift.{max u w, v} (Cardinal.mk E)) (Max.max (Max.max (Cardinal. …
  -/
  let K := Algebra.adjoin F (Set.range x)
  /-
    F : Type u
    E : Type v
    inst✝⁵ : CommRing F
    inst✝⁴ : Nontrivial F
    inst✝³ : CommRing E
    inst✝² : IsDomain E
    inst✝¹ : Algebra F E
    ι : Type w
    x : ι → E
    inst✝ : Nonempty ι
    hx : IsTranscendenceBasis F x
    K : Subalgebra F E := Algebra.adjoin F (Set.range x)
    ⊢ Eq (Cardinal.lift.{max u w, v} (Cardinal.mk E)) (Max.max (Max.max (Cardinal. …
  -/
  suffices #E = #K by simp [K, this, ← lift_mk_eq'.2 ⟨hx.1.aevalEquiv.toEquiv⟩]
  /-
    F : Type u
    E : Type v
    inst✝⁵ : CommRing F
    inst✝⁴ : Nontrivial F
    inst✝³ : CommRing E
    inst✝² : IsDomain E
    inst✝¹ : Algebra F E
    ι : Type w
    x : ι → E
    inst✝ : Nonempty ι
    hx : IsTranscendenceBasis F x
    K : Subalgebra F E := Algebra.adjoin F (Set.range x)
    ⊢ Eq (Cardinal.mk E) (Cardinal.mk (Subtype fun x => Membership.mem K x))
  -/
  haveI : Algebra.IsAlgebraic K E := hx.isAlgebraic
  /-
    F : Type u
    E : Type v
    inst✝⁵ : CommRing F
    inst✝⁴ : Nontrivial F
    inst✝³ : CommRing E
    inst✝² : IsDomain E
    inst✝¹ : Algebra F E
    ι : Type w
    x : ι → E
    inst✝ : Nonempty ι
    hx : IsTranscendenceBasis F x
    K : Subalgebra F E := Algebra.adjoin F (Set.range x)
    this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem K x) E
    ⊢ Eq (Cardinal.mk E) (Cardinal.mk (Subtype fun x => Membership.mem K x))
  -/
  refine le_antisymm ?_ (mk_le_of_injective Subtype.val_injective)
  /-
    F : Type u
    E : Type v
    inst✝⁵ : CommRing F
    inst✝⁴ : Nontrivial F
    inst✝³ : CommRing E
    inst✝² : IsDomain E
    inst✝¹ : Algebra F E
    ι : Type w
    x : ι → E
    inst✝ : Nonempty ι
    hx : IsTranscendenceBasis F x
    K : Subalgebra F E := Algebra.adjoin F (Set.range x)
    this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem K x) E
    ⊢ LE.le (Cardinal.mk E) (Cardinal.mk (Subtype fun x => Membership.mem K x))
  -/
  haveI : Infinite K := hx.1.aevalEquiv.infinite_iff.1 inferInstance
  /-
    F : Type u
    E : Type v
    inst✝⁵ : CommRing F
    inst✝⁴ : Nontrivial F
    inst✝³ : CommRing E
    inst✝² : IsDomain E
    inst✝¹ : Algebra F E
    ι : Type w
    x : ι → E
    inst✝ : Nonempty ι
    hx : IsTranscendenceBasis F x
    K : Subalgebra F E := Algebra.adjoin F (Set.range x)
    this✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem K x) E
    this : Infinite (Subtype fun x => Membership.mem K x)
    ⊢ LE.le (Cardinal.mk E) (Cardinal.mk (Subtype fun x => Membership.mem K x))
  -/
  simpa only [sup_eq_left.2 (aleph0_le_mk K)] using Algebra.IsAlgebraic.cardinalMk_le_max K E
  /-
    🎉 no goals
  -/


theorem IsTranscendenceBasis.lift_rank_eq_max_lift
    {F : Type u} {E : Type v} [Field F] [Field E] [Algebra F E]
    {ι : Type w} {x : ι → E} [Nonempty ι] (hx : IsTranscendenceBasis F x) :
    lift.{max u w} (Module.rank F E) = lift.{max v w} #F ⊔ lift.{max u v} #ι ⊔ ℵ₀ := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    ι : Type w
    x : ι → E
    inst✝ : Nonempty ι
    hx : IsTranscendenceBasis F x
    ⊢ Eq (Cardinal.lift.{max u w, v} (Module.rank F E)) (Max.max (Max.max (Cardina …
  -/
  let K := IntermediateField.adjoin F (Set.range x)
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    ι : Type w
    x : ι → E
    inst✝ : Nonempty ι
    hx : IsTranscendenceBasis F x
    K : IntermediateField F E := IntermediateField.adjoin F (Set.range x)
    ⊢ Eq (Cardinal.lift.{max u w, v} (Module.rank F E)) (Max.max (Max.max (Cardina …
  -/
  haveI : Algebra.IsAlgebraic K E := hx.isAlgebraic_field
  rw [← rank_mul_rank F K E, lift_mul, ← hx.1.aevalEquivField.toLinearEquiv.lift_rank_eq,
    MvRatFunc.rank_eq_max_lift, lift_max, lift_max, lift_lift, lift_lift, lift_aleph0]
  refine mul_eq_left le_sup_right ((lift_le.2 ((rank_le_card K E).trans
    (Algebra.IsAlgebraic.cardinalMk_le_max K E))).trans_eq ?_) (by simp [rank_pos.ne'])
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    ι : Type w
    x : ι → E
    inst✝ : Nonempty ι
    hx : IsTranscendenceBasis F x
    K : IntermediateField F E := IntermediateField.adjoin F (Set.range x)
    this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem K x) E
    ⊢ Eq (Cardinal.lift.{max u w, v} (Max.max (Cardinal.mk (Subtype fun x => Membe …
  -/
  simp [K, ← lift_mk_eq'.2 ⟨hx.1.aevalEquivField.toEquiv⟩]
  /-
    🎉 no goals
  -/


theorem Algebra.Transcendental.rank_eq_cardinalMk
    (F : Type u) (E : Type v) [Field F] [Field E] [Algebra F E] [Algebra.Transcendental F E] :
    Module.rank F E = #E := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.Transcendental F E
    ⊢ Eq (Module.rank F E) (Cardinal.mk E)
  -/
  obtain ⟨ι, x, hx⟩ := exists_isTranscendenceBasis' _ (algebraMap F E).injective
  /-
    case intro.intro
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.Transcendental F E
    ι : Type v
    x : ι → E
    hx : IsTranscendenceBasis F x
    ⊢ Eq (Module.rank F E) (Cardinal.mk E)
  -/
  haveI := hx.nonempty_iff_transcendental.2 ‹_›
  /-
    case intro.intro
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.Transcendental F E
    ι : Type v
    x : ι → E
    hx : IsTranscendenceBasis F x
    this : Nonempty ι
    ⊢ Eq (Module.rank F E) (Cardinal.mk E)
  -/
  simpa [← hx.lift_cardinalMk_eq_max_lift] using hx.lift_rank_eq_max_lift
  /-
    🎉 no goals
  -/


theorem IntermediateField.rank_sup_le
    {F : Type u} {E : Type v} [Field F] [Field E] [Algebra F E] (A B : IntermediateField F E) :
    Module.rank F ↥(A ⊔ B) ≤ Module.rank F A * Module.rank F B := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    ⊢ LE.le (Module.rank F (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
  -/
  by_cases hA : Algebra.IsAlgebraic F A
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      A B : IntermediateField F E
      hA : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem A x)
      ⊢ LE.le (Module.rank F (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
    -/
  · exact rank_sup_le_of_isAlgebraic A B (Or.inl hA)
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    hA : Not (Algebra.IsAlgebraic F (Subtype fun x => Membership.mem A x))
    ⊢ LE.le (Module.rank F (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
  -/
  by_cases hB : Algebra.IsAlgebraic F B
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      A B : IntermediateField F E
      hA : Not (Algebra.IsAlgebraic F (Subtype fun x => Membership.mem A x))
      hB : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem B x)
      ⊢ LE.le (Module.rank F (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
    -/
  · exact rank_sup_le_of_isAlgebraic A B (Or.inr hB)
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    hA : Not (Algebra.IsAlgebraic F (Subtype fun x => Membership.mem A x))
    hB : Not (Algebra.IsAlgebraic F (Subtype fun x => Membership.mem B x))
    ⊢ LE.le (Module.rank F (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
  -/
  rw [← Algebra.transcendental_iff_not_isAlgebraic] at hA hB
  haveI : Algebra.Transcendental F ↥(A ⊔ B) := .ringHom_of_comp_eq (RingHom.id F)
    (inclusion le_sup_left) Function.surjective_id (inclusion_injective _) rfl
  /-
    case neg
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    hA : Algebra.Transcendental F (Subtype fun x => Membership.mem A x)
    hB : Algebra.Transcendental F (Subtype fun x => Membership.mem B x)
    this : Algebra.Transcendental F (Subtype fun x => Membership.mem (Max.max A B) …
    ⊢ LE.le (Module.rank F (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
  -/
  haveI := Algebra.Transcendental.infinite F A
  /-
    case neg
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    hA : Algebra.Transcendental F (Subtype fun x => Membership.mem A x)
    hB : Algebra.Transcendental F (Subtype fun x => Membership.mem B x)
    this✝ : Algebra.Transcendental F (Subtype fun x => Membership.mem (Max.max A B …
    this : Infinite (Subtype fun x => Membership.mem A x)
    ⊢ LE.le (Module.rank F (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
  -/
  haveI := Algebra.Transcendental.infinite F B
  /-
    case neg
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    hA : Algebra.Transcendental F (Subtype fun x => Membership.mem A x)
    hB : Algebra.Transcendental F (Subtype fun x => Membership.mem B x)
    this✝¹ : Algebra.Transcendental F (Subtype fun x => Membership.mem (Max.max A  …
    this✝ : Infinite (Subtype fun x => Membership.mem A x)
    this : Infinite (Subtype fun x => Membership.mem B x)
    ⊢ LE.le (Module.rank F (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
  -/
  simp_rw [Algebra.Transcendental.rank_eq_cardinalMk]
  /-
    case neg
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    hA : Algebra.Transcendental F (Subtype fun x => Membership.mem A x)
    hB : Algebra.Transcendental F (Subtype fun x => Membership.mem B x)
    this✝¹ : Algebra.Transcendental F (Subtype fun x => Membership.mem (Max.max A  …
    this✝ : Infinite (Subtype fun x => Membership.mem A x)
    this : Infinite (Subtype fun x => Membership.mem B x)
    ⊢ LE.le (Cardinal.mk (Subtype fun x => Membership.mem (Max.max A B) x)) (HMul. …
  -/
  rw [sup_def, mul_mk_eq_max, ← Cardinal.lift_le.{u}]
  /-
    case neg
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    hA : Algebra.Transcendental F (Subtype fun x => Membership.mem A x)
    hB : Algebra.Transcendental F (Subtype fun x => Membership.mem B x)
    this✝¹ : Algebra.Transcendental F (Subtype fun x => Membership.mem (Max.max A  …
    this✝ : Infinite (Subtype fun x => Membership.mem A x)
    this : Infinite (Subtype fun x => Membership.mem B x)
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk (Subtype fun x => Membership.mem (I …
  -/
  refine (lift_cardinalMk_adjoin_le _ _).trans ?_
  calc
    _ ≤ Cardinal.lift.{v} #F ⊔ Cardinal.lift.{u} (#A ⊔ #B) ⊔ ℵ₀ := by
      gcongr
      rw [Cardinal.lift_le]
      exact (mk_union_le _ _).trans_eq (by simp)
    _ = _ := by
      simp [lift_mk_le_lift_mk_of_injective (algebraMap F A).injective]

