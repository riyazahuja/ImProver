set_option synthInstance.maxHeartbeats 400000 in
/-- `Subfield.relrank A B` is defined to be `[B : A ⊓ B]` as a `Cardinal`, in particular,
when `A ≤ B` it is `[B : A]`, the degree of the field extension `B / A`.
This is similar to `Subgroup.relindex` but it is `Cardinal` valued. -/
noncomputable def relrank := Module.rank ↥(A ⊓ B) (extendScalars (inf_le_right : A ⊓ B ≤ B))


set_option synthInstance.maxHeartbeats 400000 in
/-- The `Nat` version of `Subfield.relrank`.
If `B / A ⊓ B` is an infinite extension, then it is zero. -/
noncomputable def relfinrank := finrank ↥(A ⊓ B) (extendScalars (inf_le_right : A ⊓ B ≤ B))


theorem relfinrank_eq_toNat_relrank : relfinrank A B = toNat (relrank A B) := rfl


theorem relrank_eq_of_inf_eq (h : A ⊓ C = B ⊓ C) : relrank A C = relrank B C := by
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h : Eq (Min.min A C) (Min.min B C)
    ⊢ Eq (A.relrank C) (B.relrank C)
  -/
  simp_rw [relrank]
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h : Eq (Min.min A C) (Min.min B C)
    ⊢ Eq (Module.rank (Subtype fun x => Membership.mem (Min.min A C) x) (Subtype f …
  -/
  congr!
  /-
    🎉 no goals
  -/


theorem relfinrank_eq_of_inf_eq (h : A ⊓ C = B ⊓ C) : relfinrank A C = relfinrank B C :=
  congr(toNat $(relrank_eq_of_inf_eq h))


set_option synthInstance.maxHeartbeats 400000 in
/-- If `A ≤ B`, then `Subfield.relrank A B` is `[B : A]` -/
theorem relrank_eq_rank_of_le (h : A ≤ B) : relrank A B = Module.rank A (extendScalars h) := by
  /-
    E : Type v
    inst✝ : Field E
    A B : Subfield E
    h : LE.le A B
    ⊢ Eq (A.relrank B) (Module.rank (Subtype fun x => Membership.mem A x) (Subtype …
  -/
  rw [relrank]
  /-
    E : Type v
    inst✝ : Field E
    A B : Subfield E
    h : LE.le A B
    ⊢ Eq (Module.rank (Subtype fun x => Membership.mem (Min.min A B) x) (Subtype f …
  -/
  have := inf_of_le_left h
  /-
    E : Type v
    inst✝ : Field E
    A B : Subfield E
    h : LE.le A B
    this : Eq (Min.min A B) A
    ⊢ Eq (Module.rank (Subtype fun x => Membership.mem (Min.min A B) x) (Subtype f …
  -/
  congr!
  /-
    🎉 no goals
  -/


set_option synthInstance.maxHeartbeats 400000 in
/-- If `A ≤ B`, then `Subfield.relfinrank A B` is `[B : A]` -/
theorem relfinrank_eq_finrank_of_le (h : A ≤ B) : relfinrank A B = finrank A (extendScalars h) :=
  congr(toNat $(relrank_eq_rank_of_le h))


theorem inf_relrank_right : relrank (A ⊓ B) B = relrank A B :=
  relrank_eq_rank_of_le (inf_le_right : A ⊓ B ≤ B)


theorem inf_relfinrank_right : relfinrank (A ⊓ B) B = relfinrank A B :=
  congr(toNat $(inf_relrank_right A B))


theorem inf_relrank_left : relrank (A ⊓ B) A = relrank B A := by
  /-
    E : Type v
    inst✝ : Field E
    A B : Subfield E
    ⊢ Eq ((Min.min A B).relrank A) (B.relrank A)
  -/
  rw [inf_comm, inf_relrank_right]
  /-
    🎉 no goals
  -/


theorem inf_relfinrank_left : relfinrank (A ⊓ B) A = relfinrank B A :=
  congr(toNat $(inf_relrank_left A B))


@[simp]
theorem relrank_self : relrank A A = 1 := by
  /-
    E : Type v
    inst✝ : Field E
    A : Subfield E
    ⊢ Eq (A.relrank A) 1
  -/
  rw [relrank_eq_rank_of_le (le_refl A), extendScalars_self, IntermediateField.rank_bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem relfinrank_self : relfinrank A A = 1 := by
  /-
    E : Type v
    inst✝ : Field E
    A : Subfield E
    ⊢ Eq (A.relfinrank A) 1
  -/
  simp [relfinrank_eq_toNat_relrank]
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relrank_eq_one_of_le (h : B ≤ A) : relrank A B = 1 := by
  /-
    E : Type v
    inst✝ : Field E
    A B : Subfield E
    h : LE.le B A
    ⊢ Eq (A.relrank B) 1
  -/
  rw [← inf_relrank_right, inf_eq_right.2 h, relrank_self]
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relfinrank_eq_one_of_le (h : B ≤ A) : relfinrank A B = 1 := by
  /-
    E : Type v
    inst✝ : Field E
    A B : Subfield E
    h : LE.le B A
    ⊢ Eq (A.relfinrank B) 1
  -/
  simp [relfinrank_eq_toNat_relrank, relrank_eq_one_of_le h]
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relrank_mul_rank_top (h : A ≤ B) : relrank A B * Module.rank B E = Module.rank A E := by
  /-
    E : Type v
    inst✝ : Field E
    A B : Subfield E
    h : LE.le A B
    ⊢ Eq (HMul.hMul (A.relrank B) (Module.rank (Subtype fun x => Membership.mem B  …
  -/
  rw [relrank_eq_rank_of_le h]
  /-
    E : Type v
    inst✝ : Field E
    A B : Subfield E
    h : LE.le A B
    ⊢ Eq (HMul.hMul (Module.rank (Subtype fun x => Membership.mem A x) (Subtype fu …
  -/
  letI : Algebra A B := (inclusion h).toAlgebra
  /-
    E : Type v
    inst✝ : Field E
    A B : Subfield E
    h : LE.le A B
    this : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Members …
    ⊢ Eq (HMul.hMul (Module.rank (Subtype fun x => Membership.mem A x) (Subtype fu …
  -/
  haveI : IsScalarTower A B E := IsScalarTower.of_algebraMap_eq' rfl
  /-
    E : Type v
    inst✝ : Field E
    A B : Subfield E
    h : LE.le A B
    this✝ : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Member …
    this : IsScalarTower (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
    ⊢ Eq (HMul.hMul (Module.rank (Subtype fun x => Membership.mem A x) (Subtype fu …
  -/
  exact rank_mul_rank A B E
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relfinrank_mul_finrank_top (h : A ≤ B) : relfinrank A B * finrank B E = finrank A E := by
  /-
    E : Type v
    inst✝ : Field E
    A B : Subfield E
    h : LE.le A B
    ⊢ Eq (HMul.hMul (A.relfinrank B) (Module.finrank (Subtype fun x => Membership. …
  -/
  simpa using congr(toNat $(relrank_mul_rank_top h))
  /-
    🎉 no goals
  -/


@[simp]
theorem relrank_top_left : relrank ⊤ A = 1 := relrank_eq_one_of_le le_top


@[simp]
theorem relfinrank_top_left : relfinrank ⊤ A = 1 := relfinrank_eq_one_of_le le_top


set_option synthInstance.maxHeartbeats 400000 in
@[simp]
theorem relrank_top_right : relrank A ⊤ = Module.rank A E := by
  rw [relrank_eq_rank_of_le (show A ≤ ⊤ from le_top), extendScalars_top,
    IntermediateField.topEquiv.toLinearEquiv.rank_eq]


@[simp]
theorem relfinrank_top_right : relfinrank A ⊤ = finrank A E := by
  /-
    E : Type v
    inst✝ : Field E
    A : Subfield E
    ⊢ Eq (A.relfinrank Top.top) (Module.finrank (Subtype fun x => Membership.mem A …
  -/
  simp [relfinrank_eq_toNat_relrank, finrank]
  /-
    🎉 no goals
  -/


theorem lift_relrank_map_map (f : E →+* L) :
    lift.{v} (relrank (A.map f) (B.map f)) = lift.{w} (relrank A B) :=
  -- typeclass inference is slow
  .symm <| Algebra.lift_rank_eq_of_equiv_equiv (((A ⊓ B).equivMapOfInjective f f.injective).trans
                           /-
                             E : Type v
                             inst✝¹ : Field E
                             L : Type w
                             inst✝ : Field L
                             A B : Subfield E
                             f : RingHom E L
                             ⊢ Eq (Subring.map f (Min.min A B).toSubring) (Min.min (Subfield.map f A) (Subf …
                           -/
    <| .subringCongr <| by rw [← map_inf]; rfl) (B.equivMapOfInjective f f.injective) rfl
                                           /-
                                             🎉 no goals
                                           -/


theorem relrank_map_map {L : Type v} [Field L] (f : E →+* L) :
    relrank (A.map f) (B.map f) = relrank A B := by
  /-
    E : Type v
    inst✝¹ : Field E
    A B : Subfield E
    L : Type v
    inst✝ : Field L
    f : RingHom E L
    ⊢ Eq ((Subfield.map f A).relrank (Subfield.map f B)) (A.relrank B)
  -/
  simpa only [lift_id] using lift_relrank_map_map A B f
  /-
    🎉 no goals
  -/


theorem lift_relrank_comap (f : L →+* E) (B : Subfield L) :
    lift.{v} (relrank (A.comap f) B) = lift.{w} (relrank A (B.map f)) :=
  (lift_relrank_map_map _ _ f).symm.trans <| congr_arg lift <| relrank_eq_of_inf_eq <| by
    /-
      E : Type v
      inst✝¹ : Field E
      L : Type w
      inst✝ : Field L
      A : Subfield E
      f : RingHom L E
      B : Subfield L
      ⊢ Eq (Min.min (Subfield.map f (Subfield.comap f A)) (Subfield.map f B)) (Min.m …
    -/
    rw [map_comap_eq, f.fieldRange_eq_map, inf_assoc, ← map_inf, top_inf_eq]
    /-
      🎉 no goals
    -/


theorem relrank_comap {L : Type v} [Field L] (f : L →+* E)
    (B : Subfield L) : relrank (A.comap f) B = relrank A (B.map f) := by
  /-
    E : Type v
    inst✝¹ : Field E
    A : Subfield E
    L : Type v
    inst✝ : Field L
    f : RingHom L E
    B : Subfield L
    ⊢ Eq ((Subfield.comap f A).relrank B) (A.relrank (Subfield.map f B))
  -/
  simpa only [lift_id] using A.lift_relrank_comap f B
  /-
    🎉 no goals
  -/


theorem relfinrank_comap (f : L →+* E) (B : Subfield L) :
    relfinrank (A.comap f) B = relfinrank A (B.map f) := by
  /-
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A : Subfield E
    f : RingHom L E
    B : Subfield L
    ⊢ Eq ((Subfield.comap f A).relfinrank B) (A.relfinrank (Subfield.map f B))
  -/
  simpa using congr(toNat $(lift_relrank_comap A f B))
  /-
    🎉 no goals
  -/


theorem lift_rank_comap (f : L →+* E) :
    lift.{v} (Module.rank (A.comap f) L) = lift.{w} (relrank A f.fieldRange) := by
  /-
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A : Subfield E
    f : RingHom L E
    ⊢ Eq (Cardinal.lift.{v, w} (Module.rank (Subtype fun x => Membership.mem (Subf …
  -/
  simpa only [relrank_top_right, ← RingHom.fieldRange_eq_map] using lift_relrank_comap A f ⊤
  /-
    🎉 no goals
  -/


theorem rank_comap {L : Type v} [Field L] (f : L →+* E) :
    Module.rank (A.comap f) L = relrank A f.fieldRange := by
  /-
    E : Type v
    inst✝¹ : Field E
    A : Subfield E
    L : Type v
    inst✝ : Field L
    f : RingHom L E
    ⊢ Eq (Module.rank (Subtype fun x => Membership.mem (Subfield.comap f A) x) L)  …
  -/
  simpa only [lift_id] using A.lift_rank_comap f
  /-
    🎉 no goals
  -/


theorem finrank_comap (f : L →+* E) : finrank (A.comap f) L = relfinrank A f.fieldRange := by
  /-
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A : Subfield E
    f : RingHom L E
    ⊢ Eq (Module.finrank (Subtype fun x => Membership.mem (Subfield.comap f A) x)  …
  -/
  simpa using congr(toNat $(lift_rank_comap A f))
  /-
    🎉 no goals
  -/


theorem relfinrank_map_map (f : E →+* L) :
    relfinrank (A.map f) (B.map f) = relfinrank A B := by
  /-
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A B : Subfield E
    f : RingHom E L
    ⊢ Eq ((Subfield.map f A).relfinrank (Subfield.map f B)) (A.relfinrank B)
  -/
  simpa using congr(toNat $(lift_relrank_map_map A B f))
  /-
    🎉 no goals
  -/


theorem lift_relrank_comap_comap_eq_lift_relrank_inf (f : L →+* E) :
    lift.{v} (relrank (A.comap f) (B.comap f)) =
    lift.{w} (relrank A (B ⊓ f.fieldRange)) := by
  /-
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A B : Subfield E
    f : RingHom L E
    ⊢ Eq (Cardinal.lift.{v, w} ((Subfield.comap f A).relrank (Subfield.comap f B)) …
  -/
  conv_lhs => rw [← lift_relrank_map_map _ _ f, map_comap_eq, map_comap_eq]
  /-
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A B : Subfield E
    f : RingHom L E
    ⊢ Eq (Cardinal.lift.{w, v} ((Min.min A f.fieldRange).relrank (Min.min B f.fiel …
  -/
  congr 1
  /-
    case e_c
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A B : Subfield E
    f : RingHom L E
    ⊢ Eq ((Min.min A f.fieldRange).relrank (Min.min B f.fieldRange)) (A.relrank (M …
  -/
  apply relrank_eq_of_inf_eq
  /-
    case e_c.h
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A B : Subfield E
    f : RingHom L E
    ⊢ Eq (Min.min (Min.min A f.fieldRange) (Min.min B f.fieldRange)) (Min.min A (M …
  -/
  rw [inf_assoc, inf_left_comm _ B, inf_of_le_left (le_refl _)]
  /-
    🎉 no goals
  -/


theorem relrank_comap_comap_eq_relrank_inf
    {L : Type v} [Field L] (f : L →+* E) :
    relrank (A.comap f) (B.comap f) = relrank A (B ⊓ f.fieldRange) := by
  /-
    E : Type v
    inst✝¹ : Field E
    A B : Subfield E
    L : Type v
    inst✝ : Field L
    f : RingHom L E
    ⊢ Eq ((Subfield.comap f A).relrank (Subfield.comap f B)) (A.relrank (Min.min B …
  -/
  simpa only [lift_id] using lift_relrank_comap_comap_eq_lift_relrank_inf A B f
  /-
    🎉 no goals
  -/


theorem relfinrank_comap_comap_eq_relfinrank_inf (f : L →+* E) :
    relfinrank (A.comap f) (B.comap f) = relfinrank A (B ⊓ f.fieldRange) := by
  /-
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A B : Subfield E
    f : RingHom L E
    ⊢ Eq ((Subfield.comap f A).relfinrank (Subfield.comap f B)) (A.relfinrank (Min …
  -/
  simpa using congr(toNat $(lift_relrank_comap_comap_eq_lift_relrank_inf A B f))
  /-
    🎉 no goals
  -/


theorem lift_relrank_comap_comap_eq_lift_relrank_of_le (f : L →+* E) (h : B ≤ f.fieldRange) :
    lift.{v} (relrank (A.comap f) (B.comap f)) =
    lift.{w} (relrank A B) := by
  /-
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A B : Subfield E
    f : RingHom L E
    h : LE.le B f.fieldRange
    ⊢ Eq (Cardinal.lift.{v, w} ((Subfield.comap f A).relrank (Subfield.comap f B)) …
  -/
  simpa only [inf_of_le_left h] using lift_relrank_comap_comap_eq_lift_relrank_inf A B f
  /-
    🎉 no goals
  -/


theorem relrank_comap_comap_eq_relrank_of_le
    {L : Type v} [Field L] (f : L →+* E) (h : B ≤ f.fieldRange) :
    relrank (A.comap f) (B.comap f) = relrank A B := by
  /-
    E : Type v
    inst✝¹ : Field E
    A B : Subfield E
    L : Type v
    inst✝ : Field L
    f : RingHom L E
    h : LE.le B f.fieldRange
    ⊢ Eq ((Subfield.comap f A).relrank (Subfield.comap f B)) (A.relrank B)
  -/
  simpa only [lift_id] using lift_relrank_comap_comap_eq_lift_relrank_of_le A B f h
  /-
    🎉 no goals
  -/


theorem relfinrank_comap_comap_eq_relfinrank_of_le (f : L →+* E) (h : B ≤ f.fieldRange) :
    relfinrank (A.comap f) (B.comap f) = relfinrank A B := by
  /-
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A B : Subfield E
    f : RingHom L E
    h : LE.le B f.fieldRange
    ⊢ Eq ((Subfield.comap f A).relfinrank (Subfield.comap f B)) (A.relfinrank B)
  -/
  simpa using congr(toNat $(lift_relrank_comap_comap_eq_lift_relrank_of_le A B f h))
  /-
    🎉 no goals
  -/


theorem lift_relrank_comap_comap_eq_lift_relrank_of_surjective
    (f : L →+* E) (h : Function.Surjective f) :
    lift.{v} (relrank (A.comap f) (B.comap f)) =
    lift.{w} (relrank A B) :=
  lift_relrank_comap_comap_eq_lift_relrank_of_le A B f fun x _ ↦ h x


theorem relrank_comap_comap_eq_relrank_of_surjective
    {L : Type v} [Field L] (f : L →+* E) (h : Function.Surjective f) :
    relrank (A.comap f) (B.comap f) = relrank A B := by
  /-
    E : Type v
    inst✝¹ : Field E
    A B : Subfield E
    L : Type v
    inst✝ : Field L
    f : RingHom L E
    h : Function.Surjective ⇑f
    ⊢ Eq ((Subfield.comap f A).relrank (Subfield.comap f B)) (A.relrank B)
  -/
  simpa using lift_relrank_comap_comap_eq_lift_relrank_of_surjective A B f h
  /-
    🎉 no goals
  -/


theorem relfinrank_comap_comap_eq_relfinrank_of_surjective
    (f : L →+* E) (h : Function.Surjective f) :
    relfinrank (A.comap f) (B.comap f) = relfinrank A B := by
  /-
    E : Type v
    inst✝¹ : Field E
    L : Type w
    inst✝ : Field L
    A B : Subfield E
    f : RingHom L E
    h : Function.Surjective ⇑f
    ⊢ Eq ((Subfield.comap f A).relfinrank (Subfield.comap f B)) (A.relfinrank B)
  -/
  simpa using congr(toNat $(lift_relrank_comap_comap_eq_lift_relrank_of_surjective A B f h))
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relrank_dvd_rank_top_of_le (h : A ≤ B) : relrank A B ∣ Module.rank A E :=
  dvd_of_mul_right_eq _ (relrank_mul_rank_top h)


variable {A B} in
theorem relfinrank_dvd_finrank_top_of_le (h : A ≤ B) : relfinrank A B ∣ finrank A E :=
  dvd_of_mul_right_eq _ (relfinrank_mul_finrank_top h)


variable {A B C} in
theorem relrank_mul_relrank (h1 : A ≤ B) (h2 : B ≤ C) :
    relrank A B * relrank B C = relrank A C := by
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h1 : LE.le A B
    h2 : LE.le B C
    ⊢ Eq (HMul.hMul (A.relrank B) (B.relrank C)) (A.relrank C)
  -/
  have h3 := h1.trans h2
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h1 : LE.le A B
    h2 : LE.le B C
    h3 : LE.le A C
    ⊢ Eq (HMul.hMul (A.relrank B) (B.relrank C)) (A.relrank C)
  -/
  rw [relrank_eq_rank_of_le h1, relrank_eq_rank_of_le h2, relrank_eq_rank_of_le h3]
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h1 : LE.le A B
    h2 : LE.le B C
    h3 : LE.le A C
    ⊢ Eq (HMul.hMul (Module.rank (Subtype fun x => Membership.mem A x) (Subtype fu …
  -/
  letI : Algebra A B := (inclusion h1).toAlgebra
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h1 : LE.le A B
    h2 : LE.le B C
    h3 : LE.le A C
    this : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Members …
    ⊢ Eq (HMul.hMul (Module.rank (Subtype fun x => Membership.mem A x) (Subtype fu …
  -/
  letI : Algebra B C := (inclusion h2).toAlgebra
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h1 : LE.le A B
    h2 : LE.le B C
    h3 : LE.le A C
    this✝ : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Member …
    this : Algebra (Subtype fun x => Membership.mem B x) (Subtype fun x => Members …
    ⊢ Eq (HMul.hMul (Module.rank (Subtype fun x => Membership.mem A x) (Subtype fu …
  -/
  letI : Algebra A C := (inclusion h3).toAlgebra
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h1 : LE.le A B
    h2 : LE.le B C
    h3 : LE.le A C
    this✝¹ : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Membe …
    this✝ : Algebra (Subtype fun x => Membership.mem B x) (Subtype fun x => Member …
    this : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Members …
    ⊢ Eq (HMul.hMul (Module.rank (Subtype fun x => Membership.mem A x) (Subtype fu …
  -/
  haveI : IsScalarTower A B C := IsScalarTower.of_algebraMap_eq' rfl
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h1 : LE.le A B
    h2 : LE.le B C
    h3 : LE.le A C
    this✝² : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Membe …
    this✝¹ : Algebra (Subtype fun x => Membership.mem B x) (Subtype fun x => Membe …
    this✝ : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Member …
    this : IsScalarTower (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
    ⊢ Eq (HMul.hMul (Module.rank (Subtype fun x => Membership.mem A x) (Subtype fu …
  -/
  exact rank_mul_rank A B C
  /-
    🎉 no goals
  -/


variable {A B C} in
theorem relfinrank_mul_relfinrank (h1 : A ≤ B) (h2 : B ≤ C) :
    relfinrank A B * relfinrank B C = relfinrank A C := by
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h1 : LE.le A B
    h2 : LE.le B C
    ⊢ Eq (HMul.hMul (A.relfinrank B) (B.relfinrank C)) (A.relfinrank C)
  -/
  simpa using congr(toNat $(relrank_mul_relrank h1 h2))
  /-
    🎉 no goals
  -/


theorem relrank_inf_mul_relrank : A.relrank (B ⊓ C) * B.relrank C = (A ⊓ B).relrank C := by
  rw [← inf_relrank_right A (B ⊓ C), ← inf_relrank_right B C, ← inf_relrank_right (A ⊓ B) C,
    inf_assoc, relrank_mul_relrank inf_le_right inf_le_right]


theorem relfinrank_inf_mul_relfinrank :
    A.relfinrank (B ⊓ C) * B.relfinrank C = (A ⊓ B).relfinrank C := by
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    ⊢ Eq (HMul.hMul (A.relfinrank (Min.min B C)) (B.relfinrank C)) ((Min.min A B). …
  -/
  simpa using congr(toNat $(relrank_inf_mul_relrank A B C))
  /-
    🎉 no goals
  -/


variable {B C} in
theorem relrank_mul_relrank_eq_inf_relrank (h : B ≤ C) :
    relrank A B * relrank B C = (A ⊓ B).relrank C := by
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h : LE.le B C
    ⊢ Eq (HMul.hMul (A.relrank B) (B.relrank C)) ((Min.min A B).relrank C)
  -/
  simpa only [inf_of_le_left h] using relrank_inf_mul_relrank A B C
  /-
    🎉 no goals
  -/


variable {B C} in
theorem relfinrank_mul_relfinrank_eq_inf_relfinrank (h : B ≤ C) :
    relfinrank A B * relfinrank B C = (A ⊓ B).relfinrank C := by
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h : LE.le B C
    ⊢ Eq (HMul.hMul (A.relfinrank B) (B.relfinrank C)) ((Min.min A B).relfinrank C)
  -/
  simpa using congr(toNat $(relrank_mul_relrank_eq_inf_relrank A h))
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relrank_inf_mul_relrank_of_le (h : A ≤ B) :
    A.relrank (B ⊓ C) * B.relrank C = A.relrank C := by
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h : LE.le A B
    ⊢ Eq (HMul.hMul (A.relrank (Min.min B C)) (B.relrank C)) (A.relrank C)
  -/
  simpa only [inf_of_le_left h] using relrank_inf_mul_relrank A B C
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relfinrank_inf_mul_relfinrank_of_le (h : A ≤ B) :
    A.relfinrank (B ⊓ C) * B.relfinrank C = A.relfinrank C := by
  /-
    E : Type v
    inst✝ : Field E
    A B C : Subfield E
    h : LE.le A B
    ⊢ Eq (HMul.hMul (A.relfinrank (Min.min B C)) (B.relfinrank C)) (A.relfinrank C)
  -/
  simpa using congr(toNat $(relrank_inf_mul_relrank_of_le C h))
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relrank_dvd_of_le_left (h : A ≤ B) : B.relrank C ∣ A.relrank C :=
  dvd_of_mul_left_eq _ (relrank_inf_mul_relrank_of_le C h)


variable {A B} in
theorem relfinrank_dvd_of_le_left (h : A ≤ B) : B.relfinrank C ∣ A.relfinrank C :=
  dvd_of_mul_left_eq _ (relfinrank_inf_mul_relfinrank_of_le C h)


/-- `IntermediateField.relrank A B` is defined to be `[B : A ⊓ B]` as a `Cardinal`, in particular,
when `A ≤ B` it is `[B : A]`, the degree of the field extension `B / A`.
This is similar to `Subgroup.relindex` but it is `Cardinal` valued. -/
noncomputable def relrank := A.toSubfield.relrank B.toSubfield


/-- The `Nat` version of `IntermediateField.relrank`.
If `B / A ⊓ B` is an infinite extension, then it is zero. -/
noncomputable def relfinrank := A.toSubfield.relfinrank B.toSubfield


theorem relrank_eq_of_inf_eq (h : A ⊓ C = B ⊓ C) : relrank A C = relrank B C :=
  Subfield.relrank_eq_of_inf_eq congr(toSubfield $h)


/-- If `A ≤ B`, then `IntermediateField.relrank A B` is `[B : A]` -/
theorem relrank_eq_rank_of_le (h : A ≤ B) : relrank A B = Module.rank A (extendScalars h) :=
  Subfield.relrank_eq_rank_of_le h


/-- If `A ≤ B`, then `IntermediateField.relrank A B` is `[B : A]` -/
theorem relfinrank_eq_finrank_of_le (h : A ≤ B) : relfinrank A B = finrank A (extendScalars h) :=
  congr(toNat $(relrank_eq_rank_of_le h))


theorem inf_relrank_left : relrank (A ⊓ B) A = relrank B A := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    ⊢ Eq ((Min.min A B).relrank A) (B.relrank A)
  -/
  rw [inf_comm, inf_relrank_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem relrank_self : relrank A A = 1 := A.toSubfield.relrank_self


@[simp]
theorem relfinrank_self : relfinrank A A = 1 := A.toSubfield.relfinrank_self


variable {A B} in
theorem relrank_eq_one_of_le (h : B ≤ A) : relrank A B = 1 := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    h : LE.le B A
    ⊢ Eq (A.relrank B) 1
  -/
  rw [← inf_relrank_right, inf_eq_right.2 h, relrank_self]
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relfinrank_eq_one_of_le (h : B ≤ A) : relfinrank A B = 1 := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    h : LE.le B A
    ⊢ Eq (A.relfinrank B) 1
  -/
  simp [relfinrank_eq_toNat_relrank, relrank_eq_one_of_le h]
  /-
    🎉 no goals
  -/


theorem lift_rank_comap (f : L →ₐ[F] E) :
    Cardinal.lift.{v} (Module.rank (A.comap f) L) = Cardinal.lift.{w} (relrank A f.fieldRange) :=
  A.toSubfield.lift_rank_comap f.toRingHom


theorem rank_comap {L : Type v} [Field L] [Algebra F L] (f : L →ₐ[F] E) :
    Module.rank (A.comap f) L = relrank A f.fieldRange := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    A : IntermediateField F E
    L : Type v
    inst✝¹ : Field L
    inst✝ : Algebra F L
    f : AlgHom F L E
    ⊢ Eq (Module.rank (Subtype fun x => Membership.mem (IntermediateField.comap f  …
  -/
  simpa only [lift_id] using A.lift_rank_comap f
  /-
    🎉 no goals
  -/


theorem finrank_comap (f : L →ₐ[F] E) : finrank (A.comap f) L = relfinrank A f.fieldRange := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    L : Type w
    inst✝¹ : Field L
    inst✝ : Algebra F L
    A : IntermediateField F E
    f : AlgHom F L E
    ⊢ Eq (Module.finrank (Subtype fun x => Membership.mem (IntermediateField.comap …
  -/
  simpa using congr(toNat $(lift_rank_comap A f))
  /-
    🎉 no goals
  -/


theorem lift_relrank_comap (f : L →ₐ[F] E) (B : IntermediateField F L) :
    Cardinal.lift.{v} (relrank (A.comap f) B) = Cardinal.lift.{w} (relrank A (B.map f)) :=
  A.toSubfield.lift_relrank_comap f.toRingHom B.toSubfield


theorem relrank_comap {L : Type v} [Field L] [Algebra F L] (f : L →ₐ[F] E)
    (B : IntermediateField F L) : relrank (A.comap f) B = relrank A (B.map f) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    A : IntermediateField F E
    L : Type v
    inst✝¹ : Field L
    inst✝ : Algebra F L
    f : AlgHom F L E
    B : IntermediateField F L
    ⊢ Eq ((IntermediateField.comap f A).relrank B) (A.relrank (IntermediateField.m …
  -/
  simpa only [lift_id] using A.lift_relrank_comap f B
  /-
    🎉 no goals
  -/


theorem relfinrank_comap (f : L →ₐ[F] E) (B : IntermediateField F L) :
    relfinrank (A.comap f) B = relfinrank A (B.map f) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    L : Type w
    inst✝¹ : Field L
    inst✝ : Algebra F L
    A : IntermediateField F E
    f : AlgHom F L E
    B : IntermediateField F L
    ⊢ Eq ((IntermediateField.comap f A).relfinrank B) (A.relfinrank (IntermediateF …
  -/
  simpa using congr(toNat $(lift_relrank_comap A f B))
  /-
    🎉 no goals
  -/


theorem lift_relrank_map_map (f : E →ₐ[F] L) :
    Cardinal.lift.{v} (relrank (A.map f) (B.map f)) = Cardinal.lift.{w} (relrank A B) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    L : Type w
    inst✝¹ : Field L
    inst✝ : Algebra F L
    A B : IntermediateField F E
    f : AlgHom F E L
    ⊢ Eq (Cardinal.lift.{v, w} ((IntermediateField.map f A).relrank (IntermediateF …
  -/
  rw [← lift_relrank_comap, comap_map]
  /-
    🎉 no goals
  -/


theorem relrank_map_map {L : Type v} [Field L] [Algebra F L] (f : E →ₐ[F] L) :
    relrank (A.map f) (B.map f) = relrank A B := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    A B : IntermediateField F E
    L : Type v
    inst✝¹ : Field L
    inst✝ : Algebra F L
    f : AlgHom F E L
    ⊢ Eq ((IntermediateField.map f A).relrank (IntermediateField.map f B)) (A.relr …
  -/
  simpa only [lift_id] using lift_relrank_map_map A B f
  /-
    🎉 no goals
  -/


theorem relfinrank_map_map (f : E →ₐ[F] L) :
    relfinrank (A.map f) (B.map f) = relfinrank A B := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    L : Type w
    inst✝¹ : Field L
    inst✝ : Algebra F L
    A B : IntermediateField F E
    f : AlgHom F E L
    ⊢ Eq ((IntermediateField.map f A).relfinrank (IntermediateField.map f B)) (A.r …
  -/
  simpa using congr(toNat $(lift_relrank_map_map A B f))
  /-
    🎉 no goals
  -/


theorem lift_relrank_comap_comap_eq_lift_relrank_inf (f : L →ₐ[F] E) :
    Cardinal.lift.{v} (relrank (A.comap f) (B.comap f)) =
    Cardinal.lift.{w} (relrank A (B ⊓ f.fieldRange)) :=
  A.toSubfield.lift_relrank_comap_comap_eq_lift_relrank_inf B.toSubfield f.toRingHom


theorem relrank_comap_comap_eq_relrank_inf
    {L : Type v} [Field L] [Algebra F L] (f : L →ₐ[F] E) :
    relrank (A.comap f) (B.comap f) = relrank A (B ⊓ f.fieldRange) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    A B : IntermediateField F E
    L : Type v
    inst✝¹ : Field L
    inst✝ : Algebra F L
    f : AlgHom F L E
    ⊢ Eq ((IntermediateField.comap f A).relrank (IntermediateField.comap f B)) (A. …
  -/
  simpa only [lift_id] using lift_relrank_comap_comap_eq_lift_relrank_inf A B f
  /-
    🎉 no goals
  -/


theorem relfinrank_comap_comap_eq_relfinrank_inf (f : L →ₐ[F] E) :
    relfinrank (A.comap f) (B.comap f) = relfinrank A (B ⊓ f.fieldRange) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    L : Type w
    inst✝¹ : Field L
    inst✝ : Algebra F L
    A B : IntermediateField F E
    f : AlgHom F L E
    ⊢ Eq ((IntermediateField.comap f A).relfinrank (IntermediateField.comap f B))  …
  -/
  simpa using congr(toNat $(lift_relrank_comap_comap_eq_lift_relrank_inf A B f))
  /-
    🎉 no goals
  -/


theorem lift_relrank_comap_comap_eq_lift_relrank_of_le (f : L →ₐ[F] E) (h : B ≤ f.fieldRange) :
    Cardinal.lift.{v} (relrank (A.comap f) (B.comap f)) = Cardinal.lift.{w} (relrank A B) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    L : Type w
    inst✝¹ : Field L
    inst✝ : Algebra F L
    A B : IntermediateField F E
    f : AlgHom F L E
    h : LE.le B f.fieldRange
    ⊢ Eq (Cardinal.lift.{v, w} ((IntermediateField.comap f A).relrank (Intermediat …
  -/
  simpa only [inf_of_le_left h] using lift_relrank_comap_comap_eq_lift_relrank_inf A B f
  /-
    🎉 no goals
  -/


theorem relrank_comap_comap_eq_relrank_of_le
    {L : Type v} [Field L] [Algebra F L] (f : L →ₐ[F] E) (h : B ≤ f.fieldRange) :
    relrank (A.comap f) (B.comap f) = relrank A B := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    A B : IntermediateField F E
    L : Type v
    inst✝¹ : Field L
    inst✝ : Algebra F L
    f : AlgHom F L E
    h : LE.le B f.fieldRange
    ⊢ Eq ((IntermediateField.comap f A).relrank (IntermediateField.comap f B)) (A. …
  -/
  simpa only [lift_id] using lift_relrank_comap_comap_eq_lift_relrank_of_le A B f h
  /-
    🎉 no goals
  -/


theorem relfinrank_comap_comap_eq_relfinrank_of_le (f : L →ₐ[F] E) (h : B ≤ f.fieldRange) :
    relfinrank (A.comap f) (B.comap f) = relfinrank A B := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    L : Type w
    inst✝¹ : Field L
    inst✝ : Algebra F L
    A B : IntermediateField F E
    f : AlgHom F L E
    h : LE.le B f.fieldRange
    ⊢ Eq ((IntermediateField.comap f A).relfinrank (IntermediateField.comap f B))  …
  -/
  simpa using congr(toNat $(lift_relrank_comap_comap_eq_lift_relrank_of_le A B f h))
  /-
    🎉 no goals
  -/


theorem lift_relrank_comap_comap_eq_lift_relrank_of_surjective
    (f : L →ₐ[F] E) (h : Function.Surjective f) :
    Cardinal.lift.{v} (relrank (A.comap f) (B.comap f)) = Cardinal.lift.{w} (relrank A B) :=
  lift_relrank_comap_comap_eq_lift_relrank_of_le A B f fun x _ ↦ h x


theorem relrank_comap_comap_eq_relrank_of_surjective
    {L : Type v} [Field L] [Algebra F L] (f : L →ₐ[F] E) (h : Function.Surjective f) :
    relrank (A.comap f) (B.comap f) = relrank A B := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    A B : IntermediateField F E
    L : Type v
    inst✝¹ : Field L
    inst✝ : Algebra F L
    f : AlgHom F L E
    h : Function.Surjective ⇑f
    ⊢ Eq ((IntermediateField.comap f A).relrank (IntermediateField.comap f B)) (A. …
  -/
  simpa using lift_relrank_comap_comap_eq_lift_relrank_of_surjective A B f h
  /-
    🎉 no goals
  -/


theorem relfinrank_comap_comap_eq_relfinrank_of_surjective
    (f : L →ₐ[F] E) (h : Function.Surjective f) :
    relfinrank (A.comap f) (B.comap f) = relfinrank A B := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    L : Type w
    inst✝¹ : Field L
    inst✝ : Algebra F L
    A B : IntermediateField F E
    f : AlgHom F L E
    h : Function.Surjective ⇑f
    ⊢ Eq ((IntermediateField.comap f A).relfinrank (IntermediateField.comap f B))  …
  -/
  simpa using congr(toNat $(lift_relrank_comap_comap_eq_lift_relrank_of_surjective A B f h))
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relrank_mul_rank_top (h : A ≤ B) : relrank A B * Module.rank B E = Module.rank A E :=
  Subfield.relrank_mul_rank_top h


variable {A B} in
theorem relfinrank_mul_finrank_top (h : A ≤ B) : relfinrank A B * finrank B E = finrank A E := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    h : LE.le A B
    ⊢ Eq (HMul.hMul (A.relfinrank B) (Module.finrank (Subtype fun x => Membership. …
  -/
  simpa using congr(toNat $(relrank_mul_rank_top h))
  /-
    🎉 no goals
  -/


variable {A B} in
theorem rank_bot_mul_relrank (h : A ≤ B) : Module.rank F A * relrank A B = Module.rank F B := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    h : LE.le A B
    ⊢ Eq (HMul.hMul (Module.rank F (Subtype fun x => Membership.mem A x)) (A.relra …
  -/
  rw [relrank_eq_rank_of_le h]
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    h : LE.le A B
    ⊢ Eq (HMul.hMul (Module.rank F (Subtype fun x => Membership.mem A x)) (Module. …
  -/
  letI : Algebra A B := (inclusion h).toAlgebra
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    h : LE.le A B
    this : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Members …
    ⊢ Eq (HMul.hMul (Module.rank F (Subtype fun x => Membership.mem A x)) (Module. …
  -/
  haveI : IsScalarTower F A B := IsScalarTower.of_algebraMap_eq' rfl
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    h : LE.le A B
    this✝ : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Member …
    this : IsScalarTower F (Subtype fun x => Membership.mem A x) (Subtype fun x => …
    ⊢ Eq (HMul.hMul (Module.rank F (Subtype fun x => Membership.mem A x)) (Module. …
  -/
  exact rank_mul_rank F A B
  /-
    🎉 no goals
  -/


variable {A B} in
theorem finrank_bot_mul_relfinrank (h : A ≤ B) : finrank F A * relfinrank A B = finrank F B := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    h : LE.le A B
    ⊢ Eq (HMul.hMul (Module.finrank F (Subtype fun x => Membership.mem A x)) (A.re …
  -/
  simpa using congr(toNat $(rank_bot_mul_relrank h))
  /-
    🎉 no goals
  -/


theorem relrank_dvd_rank_bot : relrank A B ∣ Module.rank F B :=
  inf_relrank_right A B ▸ dvd_of_mul_left_eq _ (rank_bot_mul_relrank inf_le_right)


theorem relfinrank_dvd_finrank_bot : relfinrank A B ∣ finrank F B :=
  inf_relfinrank_right A B ▸ dvd_of_mul_left_eq _ (finrank_bot_mul_relfinrank inf_le_right)


variable {A B C} in
theorem relrank_mul_relrank (h1 : A ≤ B) (h2 : B ≤ C) :
    relrank A B * relrank B C = relrank A C :=
  Subfield.relrank_mul_relrank h1 h2


variable {A B C} in
theorem relfinrank_mul_relfinrank (h1 : A ≤ B) (h2 : B ≤ C) :
    relfinrank A B * relfinrank B C = relfinrank A C := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B C : IntermediateField F E
    h1 : LE.le A B
    h2 : LE.le B C
    ⊢ Eq (HMul.hMul (A.relfinrank B) (B.relfinrank C)) (A.relfinrank C)
  -/
  simpa using congr(toNat $(relrank_mul_relrank h1 h2))
  /-
    🎉 no goals
  -/


theorem relrank_inf_mul_relrank : A.relrank (B ⊓ C) * B.relrank C = (A ⊓ B).relrank C :=
  Subfield.relrank_inf_mul_relrank A.toSubfield B.toSubfield C.toSubfield


theorem relfinrank_inf_mul_relfinrank :
    A.relfinrank (B ⊓ C) * B.relfinrank C = (A ⊓ B).relfinrank C := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B C : IntermediateField F E
    ⊢ Eq (HMul.hMul (A.relfinrank (Min.min B C)) (B.relfinrank C)) ((Min.min A B). …
  -/
  simpa using congr(toNat $(relrank_inf_mul_relrank A B C))
  /-
    🎉 no goals
  -/


variable {B C} in
theorem relrank_mul_relrank_eq_inf_relrank (h : B ≤ C) :
    relrank A B * relrank B C = (A ⊓ B).relrank C := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B C : IntermediateField F E
    h : LE.le B C
    ⊢ Eq (HMul.hMul (A.relrank B) (B.relrank C)) ((Min.min A B).relrank C)
  -/
  simpa only [inf_of_le_left h] using relrank_inf_mul_relrank A B C
  /-
    🎉 no goals
  -/


variable {B C} in
theorem relfinrank_mul_relfinrank_eq_inf_relfinrank (h : B ≤ C) :
    relfinrank A B * relfinrank B C = (A ⊓ B).relfinrank C := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B C : IntermediateField F E
    h : LE.le B C
    ⊢ Eq (HMul.hMul (A.relfinrank B) (B.relfinrank C)) ((Min.min A B).relfinrank C)
  -/
  simpa using congr(toNat $(relrank_mul_relrank_eq_inf_relrank A h))
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relrank_inf_mul_relrank_of_le (h : A ≤ B) :
    A.relrank (B ⊓ C) * B.relrank C = A.relrank C := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B C : IntermediateField F E
    h : LE.le A B
    ⊢ Eq (HMul.hMul (A.relrank (Min.min B C)) (B.relrank C)) (A.relrank C)
  -/
  simpa only [inf_of_le_left h] using relrank_inf_mul_relrank A B C
  /-
    🎉 no goals
  -/


variable {A B} in
theorem relfinrank_inf_mul_relfinrank_of_le (h : A ≤ B) :
    A.relfinrank (B ⊓ C) * B.relfinrank C = A.relfinrank C := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B C : IntermediateField F E
    h : LE.le A B
    ⊢ Eq (HMul.hMul (A.relfinrank (Min.min B C)) (B.relfinrank C)) (A.relfinrank C)
  -/
  simpa using congr(toNat $(relrank_inf_mul_relrank_of_le C h))
  /-
    🎉 no goals
  -/


@[simp]
theorem relrank_top_right : relrank A ⊤ = Module.rank A E := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A : IntermediateField F E
    ⊢ Eq (A.relrank Top.top) (Module.rank (Subtype fun x => Membership.mem A x) E)
  -/
  rw [← relrank_mul_rank_top (show A ≤ ⊤ from le_top), IntermediateField.rank_top, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem relfinrank_top_right : relfinrank A ⊤ = finrank A E := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A : IntermediateField F E
    ⊢ Eq (A.relfinrank Top.top) (Module.finrank (Subtype fun x => Membership.mem A …
  -/
  simp [relfinrank_eq_toNat_relrank, finrank]
  /-
    🎉 no goals
  -/


@[simp]
theorem relrank_bot_left : relrank ⊥ A = Module.rank F A := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A : IntermediateField F E
    ⊢ Eq (Bot.bot.relrank A) (Module.rank F (Subtype fun x => Membership.mem A x))
  -/
  rw [← rank_bot_mul_relrank (show ⊥ ≤ A from bot_le), IntermediateField.rank_bot, one_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem relfinrank_bot_left : relfinrank ⊥ A = finrank F A := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A : IntermediateField F E
    ⊢ Eq (Bot.bot.relfinrank A) (Module.finrank F (Subtype fun x => Membership.mem …
  -/
  simp [relfinrank_eq_toNat_relrank, finrank]
  /-
    🎉 no goals
  -/


@[simp]
theorem relrank_bot_right : relrank A ⊥ = 1 := relrank_eq_one_of_le bot_le


@[simp]
theorem relfinrank_bot_right : relfinrank A ⊥ = 1 := relfinrank_eq_one_of_le bot_le


