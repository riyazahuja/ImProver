/-- If `A` is an intermediate field of `E / F`, and `E / L / F` is a field extension tower,
then `A` and `L` are linearly disjoint, if they are linearly disjoint as subalgebras of `E`
(`Subalgebra.LinearDisjoint`). -/
protected abbrev LinearDisjoint : Prop :=
  A.toSubalgebra.LinearDisjoint (IsScalarTower.toAlgHom F L E).range


theorem linearDisjoint_iff :
    A.LinearDisjoint L ↔ A.toSubalgebra.LinearDisjoint (IsScalarTower.toAlgHom F L E).range :=
  Iff.rfl


/-- Two intermediate fields are linearly disjoint if and only if
they are linearly disjoint as subalgebras. -/
theorem linearDisjoint_iff' :
    A.LinearDisjoint B ↔ A.toSubalgebra.LinearDisjoint B.toSubalgebra := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    ⊢ Iff (A.LinearDisjoint (Subtype fun x => Membership.mem B x)) (A.LinearDisjoi …
  -/
  rw [linearDisjoint_iff]
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    ⊢ Iff (A.LinearDisjoint (IsScalarTower.toAlgHom F (Subtype fun x => Membership …
  -/
  congr!
  /-
    case a.h.e'_7
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    ⊢ Eq (IsScalarTower.toAlgHom F (Subtype fun x => Membership.mem B x) E).range  …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- Linear disjointness is symmetric. -/
theorem LinearDisjoint.symm (H : A.LinearDisjoint B) : B.LinearDisjoint A :=
  linearDisjoint_iff'.2 (linearDisjoint_iff'.1 H).symm


/-- Linear disjointness is symmetric. -/
theorem linearDisjoint_comm : A.LinearDisjoint B ↔ B.LinearDisjoint A :=
  ⟨LinearDisjoint.symm, LinearDisjoint.symm⟩


/-- Linear disjointness is symmetric. -/
theorem LinearDisjoint.symm' (H : (IsScalarTower.toAlgHom F L E).fieldRange.LinearDisjoint L') :
    (IsScalarTower.toAlgHom F L' E).fieldRange.LinearDisjoint L :=
  Subalgebra.LinearDisjoint.symm H


/-- Linear disjointness is symmetric. -/
theorem linearDisjoint_comm' :
    (IsScalarTower.toAlgHom F L E).fieldRange.LinearDisjoint L' ↔
    (IsScalarTower.toAlgHom F L' E).fieldRange.LinearDisjoint L :=
  ⟨LinearDisjoint.symm', LinearDisjoint.symm'⟩


/-- Linear disjointness of intermediate fields is preserved by algebra homomorphisms. -/
theorem map (H : A.LinearDisjoint B) {K : Type*} [Field K] [Algebra F K]
    (f : E →ₐ[F] K) : (A.map f).LinearDisjoint (B.map f) :=
  linearDisjoint_iff'.2 ((linearDisjoint_iff'.1 H).map f f.injective)


/-- Linear disjointness of an intermediate field with a tower of field embeddings is preserved by
algebra homomorphisms. -/
theorem map' (H : A.LinearDisjoint L) (K : Type*) [Field K] [Algebra F K] [Algebra L K]
    [IsScalarTower F L K] [Algebra E K] [IsScalarTower F E K] [IsScalarTower L E K] :
    (A.map (IsScalarTower.toAlgHom F E K)).LinearDisjoint L := by
  /-
    F : Type u
    E : Type v
    inst✝¹³ : Field F
    inst✝¹² : Field E
    inst✝¹¹ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra F L
    inst✝⁸ : Algebra L E
    inst✝⁷ : IsScalarTower F L E
    H : A.LinearDisjoint L
    K : Type u_1
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F K
    inst✝⁴ : Algebra L K
    inst✝³ : IsScalarTower F L K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsScalarTower L E K
    ⊢ (IntermediateField.map (IsScalarTower.toAlgHom F E K) A).LinearDisjoint L
  -/
  rw [linearDisjoint_iff] at H ⊢
  /-
    F : Type u
    E : Type v
    inst✝¹³ : Field F
    inst✝¹² : Field E
    inst✝¹¹ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra F L
    inst✝⁸ : Algebra L E
    inst✝⁷ : IsScalarTower F L E
    H : A.LinearDisjoint (IsScalarTower.toAlgHom F L E).range
    K : Type u_1
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F K
    inst✝⁴ : Algebra L K
    inst✝³ : IsScalarTower F L K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsScalarTower L E K
    ⊢ (IntermediateField.map (IsScalarTower.toAlgHom F E K) A).LinearDisjoint (IsS …
  -/
  have := H.map (IsScalarTower.toAlgHom F E K) (RingHom.injective _)
  /-
    F : Type u
    E : Type v
    inst✝¹³ : Field F
    inst✝¹² : Field E
    inst✝¹¹ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra F L
    inst✝⁸ : Algebra L E
    inst✝⁷ : IsScalarTower F L E
    H : A.LinearDisjoint (IsScalarTower.toAlgHom F L E).range
    K : Type u_1
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F K
    inst✝⁴ : Algebra L K
    inst✝³ : IsScalarTower F L K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsScalarTower L E K
    this : (Subalgebra.map (IsScalarTower.toAlgHom F E K) A.toSubalgebra).LinearDi …
    ⊢ (IntermediateField.map (IsScalarTower.toAlgHom F E K) A).LinearDisjoint (IsS …
  -/
  rw [← AlgHom.range_comp] at this
  /-
    F : Type u
    E : Type v
    inst✝¹³ : Field F
    inst✝¹² : Field E
    inst✝¹¹ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra F L
    inst✝⁸ : Algebra L E
    inst✝⁷ : IsScalarTower F L E
    H : A.LinearDisjoint (IsScalarTower.toAlgHom F L E).range
    K : Type u_1
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F K
    inst✝⁴ : Algebra L K
    inst✝³ : IsScalarTower F L K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsScalarTower L E K
    this : (Subalgebra.map (IsScalarTower.toAlgHom F E K) A.toSubalgebra).LinearDi …
    ⊢ (IntermediateField.map (IsScalarTower.toAlgHom F E K) A).LinearDisjoint (IsS …
  -/
  convert this
  /-
    case h.e'_7.h.e'_9
    F : Type u
    E : Type v
    inst✝¹³ : Field F
    inst✝¹² : Field E
    inst✝¹¹ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra F L
    inst✝⁸ : Algebra L E
    inst✝⁷ : IsScalarTower F L E
    H : A.LinearDisjoint (IsScalarTower.toAlgHom F L E).range
    K : Type u_1
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F K
    inst✝⁴ : Algebra L K
    inst✝³ : IsScalarTower F L K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsScalarTower L E K
    this : (Subalgebra.map (IsScalarTower.toAlgHom F E K) A.toSubalgebra).LinearDi …
    ⊢ Eq (IsScalarTower.toAlgHom F L K) ((IsScalarTower.toAlgHom F E K).comp (IsSc …
  -/
  ext; exact IsScalarTower.algebraMap_apply L E K _
       /-
         🎉 no goals
       -/


/-- Linear disjointness is preserved by algebra homomorphism. -/
theorem map'' {L' : Type*} [Field L'] [Algebra F L'] [Algebra L' E] [IsScalarTower F L' E]
    (H : (IsScalarTower.toAlgHom F L E).fieldRange.LinearDisjoint L')
    (K : Type*) [Field K] [Algebra F K] [Algebra L K] [IsScalarTower F L K]
    [Algebra L' K] [IsScalarTower F L' K] [Algebra E K] [IsScalarTower F E K]
    [IsScalarTower L E K] [IsScalarTower L' E K] :
    (IsScalarTower.toAlgHom F L K).fieldRange.LinearDisjoint L' := by
  /-
    F : Type u
    E : Type v
    inst✝²⁰ : Field F
    inst✝¹⁹ : Field E
    inst✝¹⁸ : Algebra F E
    L : Type w
    inst✝¹⁷ : Field L
    inst✝¹⁶ : Algebra F L
    inst✝¹⁵ : Algebra L E
    inst✝¹⁴ : IsScalarTower F L E
    L' : Type u_1
    inst✝¹³ : Field L'
    inst✝¹² : Algebra F L'
    inst✝¹¹ : Algebra L' E
    inst✝¹⁰ : IsScalarTower F L' E
    H : (IsScalarTower.toAlgHom F L E).fieldRange.LinearDisjoint L'
    K : Type u_2
    inst✝⁹ : Field K
    inst✝⁸ : Algebra F K
    inst✝⁷ : Algebra L K
    inst✝⁶ : IsScalarTower F L K
    inst✝⁵ : Algebra L' K
    inst✝⁴ : IsScalarTower F L' K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    inst✝¹ : IsScalarTower L E K
    inst✝ : IsScalarTower L' E K
    ⊢ (IsScalarTower.toAlgHom F L K).fieldRange.LinearDisjoint L'
  -/
  rw [linearDisjoint_iff] at H ⊢
  /-
    F : Type u
    E : Type v
    inst✝²⁰ : Field F
    inst✝¹⁹ : Field E
    inst✝¹⁸ : Algebra F E
    L : Type w
    inst✝¹⁷ : Field L
    inst✝¹⁶ : Algebra F L
    inst✝¹⁵ : Algebra L E
    inst✝¹⁴ : IsScalarTower F L E
    L' : Type u_1
    inst✝¹³ : Field L'
    inst✝¹² : Algebra F L'
    inst✝¹¹ : Algebra L' E
    inst✝¹⁰ : IsScalarTower F L' E
    H : (IsScalarTower.toAlgHom F L E).fieldRange.LinearDisjoint (IsScalarTower.to …
    K : Type u_2
    inst✝⁹ : Field K
    inst✝⁸ : Algebra F K
    inst✝⁷ : Algebra L K
    inst✝⁶ : IsScalarTower F L K
    inst✝⁵ : Algebra L' K
    inst✝⁴ : IsScalarTower F L' K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    inst✝¹ : IsScalarTower L E K
    inst✝ : IsScalarTower L' E K
    ⊢ (IsScalarTower.toAlgHom F L K).fieldRange.LinearDisjoint (IsScalarTower.toAl …
  -/
  have := H.map (IsScalarTower.toAlgHom F E K) (RingHom.injective _)
  /-
    F : Type u
    E : Type v
    inst✝²⁰ : Field F
    inst✝¹⁹ : Field E
    inst✝¹⁸ : Algebra F E
    L : Type w
    inst✝¹⁷ : Field L
    inst✝¹⁶ : Algebra F L
    inst✝¹⁵ : Algebra L E
    inst✝¹⁴ : IsScalarTower F L E
    L' : Type u_1
    inst✝¹³ : Field L'
    inst✝¹² : Algebra F L'
    inst✝¹¹ : Algebra L' E
    inst✝¹⁰ : IsScalarTower F L' E
    H : (IsScalarTower.toAlgHom F L E).fieldRange.LinearDisjoint (IsScalarTower.to …
    K : Type u_2
    inst✝⁹ : Field K
    inst✝⁸ : Algebra F K
    inst✝⁷ : Algebra L K
    inst✝⁶ : IsScalarTower F L K
    inst✝⁵ : Algebra L' K
    inst✝⁴ : IsScalarTower F L' K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    inst✝¹ : IsScalarTower L E K
    inst✝ : IsScalarTower L' E K
    this : (Subalgebra.map (IsScalarTower.toAlgHom F E K) (IsScalarTower.toAlgHom  …
    ⊢ (IsScalarTower.toAlgHom F L K).fieldRange.LinearDisjoint (IsScalarTower.toAl …
  -/
  simp_rw [AlgHom.fieldRange_toSubalgebra, ← AlgHom.range_comp] at this
  /-
    F : Type u
    E : Type v
    inst✝²⁰ : Field F
    inst✝¹⁹ : Field E
    inst✝¹⁸ : Algebra F E
    L : Type w
    inst✝¹⁷ : Field L
    inst✝¹⁶ : Algebra F L
    inst✝¹⁵ : Algebra L E
    inst✝¹⁴ : IsScalarTower F L E
    L' : Type u_1
    inst✝¹³ : Field L'
    inst✝¹² : Algebra F L'
    inst✝¹¹ : Algebra L' E
    inst✝¹⁰ : IsScalarTower F L' E
    H : (IsScalarTower.toAlgHom F L E).fieldRange.LinearDisjoint (IsScalarTower.to …
    K : Type u_2
    inst✝⁹ : Field K
    inst✝⁸ : Algebra F K
    inst✝⁷ : Algebra L K
    inst✝⁶ : IsScalarTower F L K
    inst✝⁵ : Algebra L' K
    inst✝⁴ : IsScalarTower F L' K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    inst✝¹ : IsScalarTower L E K
    inst✝ : IsScalarTower L' E K
    this : ((IsScalarTower.toAlgHom F E K).comp (IsScalarTower.toAlgHom F L E)).ra …
    ⊢ (IsScalarTower.toAlgHom F L K).fieldRange.LinearDisjoint (IsScalarTower.toAl …
  -/
  rw [AlgHom.fieldRange_toSubalgebra]
  /-
    F : Type u
    E : Type v
    inst✝²⁰ : Field F
    inst✝¹⁹ : Field E
    inst✝¹⁸ : Algebra F E
    L : Type w
    inst✝¹⁷ : Field L
    inst✝¹⁶ : Algebra F L
    inst✝¹⁵ : Algebra L E
    inst✝¹⁴ : IsScalarTower F L E
    L' : Type u_1
    inst✝¹³ : Field L'
    inst✝¹² : Algebra F L'
    inst✝¹¹ : Algebra L' E
    inst✝¹⁰ : IsScalarTower F L' E
    H : (IsScalarTower.toAlgHom F L E).fieldRange.LinearDisjoint (IsScalarTower.to …
    K : Type u_2
    inst✝⁹ : Field K
    inst✝⁸ : Algebra F K
    inst✝⁷ : Algebra L K
    inst✝⁶ : IsScalarTower F L K
    inst✝⁵ : Algebra L' K
    inst✝⁴ : IsScalarTower F L' K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    inst✝¹ : IsScalarTower L E K
    inst✝ : IsScalarTower L' E K
    this : ((IsScalarTower.toAlgHom F E K).comp (IsScalarTower.toAlgHom F L E)).ra …
    ⊢ (IsScalarTower.toAlgHom F L K).range.LinearDisjoint (IsScalarTower.toAlgHom  …
  -/
                         /-
                           🎉 no goals
                         -/
  convert this <;> (ext; exact IsScalarTower.algebraMap_apply _ E K _)
                         /-
                           🎉 no goals
                         -/


variable (A) in
theorem self_right : A.LinearDisjoint F := Subalgebra.LinearDisjoint.bot_right _


variable (A) in
theorem bot_right : A.LinearDisjoint (⊥ : IntermediateField F E) :=
  linearDisjoint_iff'.2 (Subalgebra.LinearDisjoint.bot_right _)


variable (F E L) in
theorem bot_left : (⊥ : IntermediateField F E).LinearDisjoint L :=
  Subalgebra.LinearDisjoint.bot_left _


/-- If `A` and `L` are linearly disjoint, then any `F`-linearly independent family on `A` remains
linearly independent over `L`. -/
theorem linearIndependent_left (H : A.LinearDisjoint L)
    {ι : Type*} {a : ι → A} (ha : LinearIndependent F a) : LinearIndependent L (A.val ∘ a) :=
  (Subalgebra.LinearDisjoint.linearIndependent_left_of_flat H ha).map_of_injective_injective
    (AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F L E)) (AddMonoidHom.id E)
        /-
          F : Type u
          E : Type v
          inst✝⁶ : Field F
          inst✝⁵ : Field E
          inst✝⁴ : Algebra F E
          A : IntermediateField F E
          L : Type w
          inst✝³ : Field L
          inst✝² : Algebra F L
          inst✝¹ : Algebra L E
          inst✝ : IsScalarTower F L E
          H : A.LinearDisjoint L
          ι : Type u_1
          a : ι → Subtype fun x => Membership.mem A x
          ha : LinearIndependent F a
          ⊢ ∀ (r : L), Eq ((AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F L E)) r) …
        -/
        /-
          🎉 no goals
        -/
                  /-
                    🎉 no goals
                  -/
    (by simp) (by simp) (fun _ _ ↦ by simp_rw [Algebra.smul_def]; rfl)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- If there exists an `F`-basis of `A` which remains linearly independent over `L`, then
`A` and `L` are linearly disjoint. -/
theorem of_basis_left {ι : Type*} (a : Basis ι F A)
    (H : LinearIndependent L (A.val ∘ a)) : A.LinearDisjoint L :=
  Subalgebra.LinearDisjoint.of_basis_left _ _ a <| H.map_of_surjective_injective
    (AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F L E)) (AddMonoidHom.id E)
                                /-
                                  F : Type u
                                  E : Type v
                                  inst✝⁶ : Field F
                                  inst✝⁵ : Field E
                                  inst✝⁴ : Algebra F E
                                  A : IntermediateField F E
                                  L : Type w
                                  inst✝³ : Field L
                                  inst✝² : Algebra F L
                                  inst✝¹ : Algebra L E
                                  inst✝ : IsScalarTower F L E
                                  ι : Type u_1
                                  a : Basis ι F (Subtype fun x => Membership.mem A x)
                                  H : LinearIndependent L (Function.comp ⇑A.val ⇑a)
                                  ⊢ ∀ (m : E), Eq ((AddMonoidHom.id E) m) 0 → Eq m 0
                                -/
                                /-
                                  🎉 no goals
                                -/
    (AlgEquiv.surjective _) (by simp) (fun _ _ ↦ by simp_rw [Algebra.smul_def]; rfl)
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- If `A` and `B` are linearly disjoint, then any `F`-linearly independent family on `B` remains
linearly independent over `A`. -/
theorem linearIndependent_right (H : A.LinearDisjoint B)
    {ι : Type*} {b : ι → B} (hb : LinearIndependent F b) : LinearIndependent A (B.val ∘ b) :=
  (linearDisjoint_iff'.1 H).linearIndependent_right_of_flat hb


/-- If there exists an `F`-basis of `B` which remains linearly independent over `A`, then
`A` and `B` are linearly disjoint. -/
theorem of_basis_right {ι : Type*} (b : Basis ι F B)
    (H : LinearIndependent A (B.val ∘ b)) : A.LinearDisjoint B :=
  linearDisjoint_iff'.2 (.of_basis_right _ _ b H)


/-- If `A` and `L` are linearly disjoint, then any `F`-linearly independent family on `L` remains
linearly independent over `A`. -/
theorem linearIndependent_right' (H : A.LinearDisjoint L) {ι : Type*} {b : ι → L}
    (hb : LinearIndependent F b) : LinearIndependent A (algebraMap L E ∘ b) := by
  apply Subalgebra.LinearDisjoint.linearIndependent_right_of_flat H <| hb.map' _
    (AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F L E)).toLinearEquiv.ker


/-- If there exists an `F`-basis of `L` which remains linearly independent over `A`, then
`A` and `L` are linearly disjoint. -/
theorem of_basis_right' {ι : Type*} (b : Basis ι F L)
    (H : LinearIndependent A (algebraMap L E ∘ b)) : A.LinearDisjoint L :=
  Subalgebra.LinearDisjoint.of_basis_right _ _
    (b.map (AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F L E)).toLinearEquiv) H


/-- If `A` and `B` are linearly disjoint, then for any `F`-linearly independent families
`{ u_i }`, `{ v_j }` of `A`, `B`, the products `{ u_i * v_j }`
are linearly independent over `F`. -/
theorem linearIndependent_mul (H : A.LinearDisjoint B) {κ ι : Type*} {a : κ → A} {b : ι → B}
    (ha : LinearIndependent F a) (hb : LinearIndependent F b) :
    LinearIndependent F fun (i : κ × ι) ↦ (a i.1).1 * (b i.2).1 :=
  (linearDisjoint_iff'.1 H).linearIndependent_mul_of_flat_left ha hb


/-- If `A` and `L` are linearly disjoint, then for any `F`-linearly independent families
`{ u_i }`, `{ v_j }` of `A`, `L`, the products `{ u_i * v_j }`
are linearly independent over `F`. -/
theorem linearIndependent_mul' (H : A.LinearDisjoint L) {κ ι : Type*} {a : κ → A} {b : ι → L}
    (ha : LinearIndependent F a) (hb : LinearIndependent F b) :
    LinearIndependent F fun (i : κ × ι) ↦ (a i.1).1 * algebraMap L E (b i.2) := by
  apply Subalgebra.LinearDisjoint.linearIndependent_mul_of_flat_left H ha <| hb.map' _
    (AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F L E)).toLinearEquiv.ker


/-- If there are `F`-bases `{ u_i }`, `{ v_j }` of `A`, `B`, such that the products
`{ u_i * v_j }` are linearly independent over `F`, then `A` and `B` are linearly disjoint. -/
theorem of_basis_mul {κ ι : Type*} (a : Basis κ F A) (b : Basis ι F B)
    (H : LinearIndependent F fun (i : κ × ι) ↦ (a i.1).1 * (b i.2).1) : A.LinearDisjoint B :=
  linearDisjoint_iff'.2 (.of_basis_mul _ _ a b H)


/-- If there are `F`-bases `{ u_i }`, `{ v_j }` of `A`, `L`, such that the products
`{ u_i * v_j }` are linearly independent over `F`, then `A` and `L` are linearly disjoint. -/
theorem of_basis_mul' {κ ι : Type*} (a : Basis κ F A) (b : Basis ι F L)
    (H : LinearIndependent F fun (i : κ × ι) ↦ (a i.1).1 * algebraMap L E (b i.2)) :
    A.LinearDisjoint L :=
  Subalgebra.LinearDisjoint.of_basis_mul _ _ a
    (b.map (AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F L E)).toLinearEquiv) H


theorem of_le_left {A' : IntermediateField F E} (H : A.LinearDisjoint L)
    (h : A' ≤ A) : A'.LinearDisjoint L :=
  Subalgebra.LinearDisjoint.of_le_left_of_flat H h


theorem of_le_right {B' : IntermediateField F E} (H : A.LinearDisjoint B)
    (h : B' ≤ B) : A.LinearDisjoint B' :=
  linearDisjoint_iff'.2 ((linearDisjoint_iff'.1 H).of_le_right_of_flat h)


/-- Similar to `IntermediateField.LinearDisjoint.of_le_right` but this is for abstract fields. -/
theorem of_le_right' (H : A.LinearDisjoint L) (L' : Type*) [Field L']
    [Algebra F L'] [Algebra L' L] [IsScalarTower F L' L]
    [Algebra L' E] [IsScalarTower F L' E] [IsScalarTower L' L E] : A.LinearDisjoint L' := by
  /-
    F : Type u
    E : Type v
    inst✝¹³ : Field F
    inst✝¹² : Field E
    inst✝¹¹ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra F L
    inst✝⁸ : Algebra L E
    inst✝⁷ : IsScalarTower F L E
    H : A.LinearDisjoint L
    L' : Type u_1
    inst✝⁶ : Field L'
    inst✝⁵ : Algebra F L'
    inst✝⁴ : Algebra L' L
    inst✝³ : IsScalarTower F L' L
    inst✝² : Algebra L' E
    inst✝¹ : IsScalarTower F L' E
    inst✝ : IsScalarTower L' L E
    ⊢ A.LinearDisjoint L'
  -/
  refine Subalgebra.LinearDisjoint.of_le_right_of_flat H ?_
  /-
    F : Type u
    E : Type v
    inst✝¹³ : Field F
    inst✝¹² : Field E
    inst✝¹¹ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra F L
    inst✝⁸ : Algebra L E
    inst✝⁷ : IsScalarTower F L E
    H : A.LinearDisjoint L
    L' : Type u_1
    inst✝⁶ : Field L'
    inst✝⁵ : Algebra F L'
    inst✝⁴ : Algebra L' L
    inst✝³ : IsScalarTower F L' L
    inst✝² : Algebra L' E
    inst✝¹ : IsScalarTower F L' E
    inst✝ : IsScalarTower L' L E
    ⊢ LE.le (IsScalarTower.toAlgHom F L' E).range (IsScalarTower.toAlgHom F L E).r …
  -/
  convert AlgHom.range_comp_le_range (IsScalarTower.toAlgHom F L' L) (IsScalarTower.toAlgHom F L E)
  /-
    case h.e'_3.h.e'_9
    F : Type u
    E : Type v
    inst✝¹³ : Field F
    inst✝¹² : Field E
    inst✝¹¹ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra F L
    inst✝⁸ : Algebra L E
    inst✝⁷ : IsScalarTower F L E
    H : A.LinearDisjoint L
    L' : Type u_1
    inst✝⁶ : Field L'
    inst✝⁵ : Algebra F L'
    inst✝⁴ : Algebra L' L
    inst✝³ : IsScalarTower F L' L
    inst✝² : Algebra L' E
    inst✝¹ : IsScalarTower F L' E
    inst✝ : IsScalarTower L' L E
    ⊢ Eq (IsScalarTower.toAlgHom F L' E) ((IsScalarTower.toAlgHom F L E).comp (IsS …
  -/
  ext; exact IsScalarTower.algebraMap_apply L' L E _
       /-
         🎉 no goals
       -/


/-- If `A` and `B` are linearly disjoint, `A'` and `B'` are contained in `A` and `B`,
respectively, then `A'` and `B'` are also linearly disjoint. -/
theorem of_le {A' B' : IntermediateField F E} (H : A.LinearDisjoint B)
    (hA : A' ≤ A) (hB : B' ≤ B) : A'.LinearDisjoint B' :=
  H.of_le_left hA |>.of_le_right hB


/-- Similar to `IntermediateField.LinearDisjoint.of_le` but this is for abstract fields. -/
theorem of_le' {A' : IntermediateField F E} (H : A.LinearDisjoint L)
    (hA : A' ≤ A) (L' : Type*) [Field L']
    [Algebra F L'] [Algebra L' L] [IsScalarTower F L' L]
    [Algebra L' E] [IsScalarTower F L' E] [IsScalarTower L' L E] : A'.LinearDisjoint L' :=
  H.of_le_left hA |>.of_le_right' L'


/-- If `A` and `B` are linearly disjoint over `F`, then their intersection is equal to `F`. -/
theorem inf_eq_bot (H : A.LinearDisjoint B) :
    A ⊓ B = ⊥ := toSubalgebra_injective (linearDisjoint_iff'.1 H).inf_eq_bot


/-- If `A` and `A` itself are linearly disjoint over `F`, then it is equal to `F`. -/
theorem eq_bot_of_self (H : A.LinearDisjoint A) : A = ⊥ :=
  inf_idem A ▸ H.inf_eq_bot


/-- If `A` and `B` are linearly disjoint over `F`, then the
rank of `A ⊔ B` is equal to the product of that of `A` and `B`. -/
theorem rank_sup (H : A.LinearDisjoint B) :
    Module.rank F ↥(A ⊔ B) = Module.rank F A * Module.rank F B :=
  have h := le_sup_toSubalgebra A B
  (rank_sup_le A B).antisymm <|
    (linearDisjoint_iff'.1 H).rank_sup_of_free.ge.trans <|
      (Subalgebra.inclusion h).toLinearMap.rank_le_of_injective (Subalgebra.inclusion_injective h)


/-- If `A` and `B` are linearly disjoint over `F`, then the `Module.finrank` of
`A ⊔ B` is equal to the product of that of `A` and `B`. -/
theorem finrank_sup (H : A.LinearDisjoint B) : finrank F ↥(A ⊔ B) = finrank F A * finrank F B := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    A B : IntermediateField F E
    H : A.LinearDisjoint (Subtype fun x => Membership.mem B x)
    ⊢ Eq (Module.finrank F (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
  -/
  simpa only [map_mul] using congr(Cardinal.toNat $(H.rank_sup))
  /-
    🎉 no goals
  -/


/-- If `A` and `B` are finite extensions of `F`,
such that rank of `A ⊔ B` is equal to the product of the rank of `A` and `B`,
then `A` and `B` are linearly disjoint. -/
theorem of_finrank_sup [FiniteDimensional F A] [FiniteDimensional F B]
    (H : finrank F ↥(A ⊔ B) = finrank F A * finrank F B) : A.LinearDisjoint B :=
                                                       /-
                                                         F : Type u
                                                         E : Type v
                                                         inst✝⁴ : Field F
                                                         inst✝³ : Field E
                                                         inst✝² : Algebra F E
                                                         A B : IntermediateField F E
                                                         inst✝¹ : FiniteDimensional F (Subtype fun x => Membership.mem A x)
                                                         inst✝ : FiniteDimensional F (Subtype fun x => Membership.mem B x)
                                                         H : Eq (Module.finrank F (Subtype fun x => Membership.mem (Max.max A B) x)) (H …
                                                         ⊢ Eq (Module.finrank F (Subtype fun x => Membership.mem (Max.max A.toSubalgebr …
                                                       -/
  linearDisjoint_iff'.2 <| .of_finrank_sup_of_free (by rwa [← sup_toSubalgebra_of_left])
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- If `A` and `L` have coprime degree over `F`, then they are linearly disjoint. -/
theorem of_finrank_coprime (H : (finrank F A).Coprime (finrank F L)) : A.LinearDisjoint L :=
  letI : Field (AlgHom.range (IsScalarTower.toAlgHom F L E)) :=
    inferInstanceAs <| Field (AlgHom.fieldRange (IsScalarTower.toAlgHom F L E))
  letI : Field A.toSubalgebra := inferInstanceAs <| Field A
  Subalgebra.LinearDisjoint.of_finrank_coprime_of_free <| by
    /-
      F : Type u
      E : Type v
      inst✝⁶ : Field F
      inst✝⁵ : Field E
      inst✝⁴ : Algebra F E
      A : IntermediateField F E
      L : Type w
      inst✝³ : Field L
      inst✝² : Algebra F L
      inst✝¹ : Algebra L E
      inst✝ : IsScalarTower F L E
      H : (Module.finrank F (Subtype fun x => Membership.mem A x)).Coprime (Module.f …
      this✝ : Field (Subtype fun x => Membership.mem (IsScalarTower.toAlgHom F L E). …
      this : Field (Subtype fun x => Membership.mem A.toSubalgebra x) := inferInstan …
      ⊢ (Module.finrank F (Subtype fun x => Membership.mem A.toSubalgebra x)).Coprim …
    -/
    rwa [(AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F L E)).toLinearEquiv.finrank_eq] at H
    /-
      🎉 no goals
    -/


/-- If `A` and `L` are linearly disjoint over `F`, then `A ⊗[F] L` is a domain. -/
theorem isDomain (H : A.LinearDisjoint L) : IsDomain (A ⊗[F] L) :=
  have : IsDomain (A ⊗[F] _) := Subalgebra.LinearDisjoint.isDomain H
  (Algebra.TensorProduct.congr (AlgEquiv.refl : A ≃ₐ[F] A)
    (AlgEquiv.ofInjective (IsScalarTower.toAlgHom F L E) (RingHom.injective _))).toMulEquiv.isDomain


/-- If `A` and `B` are field extensions of `F`, there exists a field extension `E` of `F` that
`A` and `B` embed into with linearly disjoint images, then `A ⊗[F] B` is a domain. -/
theorem isDomain' {A B : Type*} [Field A] [Algebra F A] [Field B] [Algebra F B]
    {fa : A →ₐ[F] E} {fb : B →ₐ[F] E} (H : fa.fieldRange.LinearDisjoint fb.fieldRange) :
    IsDomain (A ⊗[F] B) := by
  /-
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Algebra F A
    inst✝¹ : Field B
    inst✝ : Algebra F B
    fa : AlgHom F A E
    fb : AlgHom F B E
    H : fa.fieldRange.LinearDisjoint (Subtype fun x => Membership.mem fb.fieldRang …
    ⊢ IsDomain (TensorProduct F A B)
  -/
  simp_rw [linearDisjoint_iff', AlgHom.fieldRange_toSubalgebra] at H
  /-
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Algebra F A
    inst✝¹ : Field B
    inst✝ : Algebra F B
    fa : AlgHom F A E
    fb : AlgHom F B E
    H : fa.range.LinearDisjoint fb.range
    ⊢ IsDomain (TensorProduct F A B)
  -/
  exact H.isDomain_of_injective fa.injective fb.injective
  /-
    🎉 no goals
  -/


/-- If `A ⊗[F] L` is a field, then `A` and `L` are linearly disjoint over `F`. -/
theorem of_isField (H : IsField (A ⊗[F] L)) : A.LinearDisjoint L := by
  /-
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra L E
    inst✝ : IsScalarTower F L E
    H : IsField (TensorProduct F (Subtype fun x => Membership.mem A x) L)
    ⊢ A.LinearDisjoint L
  -/
  apply Subalgebra.LinearDisjoint.of_isField
  -- need these otherwise the `exact` will stuck at typeclass
  /-
    case H
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra L E
    inst✝ : IsScalarTower F L E
    H : IsField (TensorProduct F (Subtype fun x => Membership.mem A x) L)
    ⊢ IsField (TensorProduct F (Subtype fun x => Membership.mem A.toSubalgebra x)  …
  -/
  haveI : SMulCommClass F A A := SMulCommClass.of_commMonoid F A A
  /-
    case H
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra L E
    inst✝ : IsScalarTower F L E
    H : IsField (TensorProduct F (Subtype fun x => Membership.mem A x) L)
    this : SMulCommClass F (Subtype fun x => Membership.mem A x) (Subtype fun x => …
    ⊢ IsField (TensorProduct F (Subtype fun x => Membership.mem A.toSubalgebra x)  …
  -/
  haveI : SMulCommClass F A.toSubalgebra A.toSubalgebra := ‹SMulCommClass F A A›
  /-
    case H
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    A : IntermediateField F E
    L : Type w
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra L E
    inst✝ : IsScalarTower F L E
    H : IsField (TensorProduct F (Subtype fun x => Membership.mem A x) L)
    this✝ : SMulCommClass F (Subtype fun x => Membership.mem A x) (Subtype fun x = …
    this : SMulCommClass F (Subtype fun x => Membership.mem A.toSubalgebra x) (Sub …
    ⊢ IsField (TensorProduct F (Subtype fun x => Membership.mem A.toSubalgebra x)  …
  -/
  letI : Mul (A ⊗[F] L) := Algebra.TensorProduct.instMul
  letI : Mul (A.toSubalgebra ⊗[F] (IsScalarTower.toAlgHom F L E).range) :=
    Algebra.TensorProduct.instMul
  exact Algebra.TensorProduct.congr (AlgEquiv.refl : A ≃ₐ[F] A)
    (AlgEquiv.ofInjective (IsScalarTower.toAlgHom F L E) (RingHom.injective _))
      |>.symm.toMulEquiv.isField _ H


/-- If `A` and `B` are field extensions of `F`, such that `A ⊗[F] B` is a field, then for any
field extension of `F` that `A` and `B` embed into, their images are linearly disjoint. -/
theorem of_isField' {A : Type v} [Field A] {B : Type w} [Field B]
    [Algebra F A] [Algebra F B] (H : IsField (A ⊗[F] B))
    {K : Type*} [Field K] [Algebra F K] (fa : A →ₐ[F] K) (fb : B →ₐ[F] K) :
    fa.fieldRange.LinearDisjoint fb.fieldRange := by
  /-
    F : Type u
    inst✝⁶ : Field F
    A : Type v
    inst✝⁵ : Field A
    B : Type w
    inst✝⁴ : Field B
    inst✝³ : Algebra F A
    inst✝² : Algebra F B
    H : IsField (TensorProduct F A B)
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Algebra F K
    fa : AlgHom F A K
    fb : AlgHom F B K
    ⊢ fa.fieldRange.LinearDisjoint (Subtype fun x => Membership.mem fb.fieldRange x)
  -/
  rw [linearDisjoint_iff']
  /-
    F : Type u
    inst✝⁶ : Field F
    A : Type v
    inst✝⁵ : Field A
    B : Type w
    inst✝⁴ : Field B
    inst✝³ : Algebra F A
    inst✝² : Algebra F B
    H : IsField (TensorProduct F A B)
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Algebra F K
    fa : AlgHom F A K
    fb : AlgHom F B K
    ⊢ fa.fieldRange.LinearDisjoint fb.fieldRange.toSubalgebra
  -/
  apply Subalgebra.LinearDisjoint.of_isField
  exact Algebra.TensorProduct.congr (AlgEquiv.ofInjective fa fa.injective)
    (AlgEquiv.ofInjective fb fb.injective) |>.symm.toMulEquiv.isField _ H


variable (F) in
/-- If `A` and `B` are field extensions of `F`, such that `A ⊗[F] B` is a domain, then there exists
a field extension of `F` that `A` and `B` embed into with linearly disjoint images. -/
theorem exists_field_of_isDomain (A : Type v) [Field A] (B : Type w) [Field B]
    [Algebra F A] [Algebra F B] [IsDomain (A ⊗[F] B)] :
    ∃ (K : Type (max v w)) (_ : Field K) (_ : Algebra F K) (fa : A →ₐ[F] K) (fb : B →ₐ[F] K),
    fa.fieldRange.LinearDisjoint fb.fieldRange :=
  have ⟨K, inst1, inst2, fa, fb, _, _, H⟩ :=
    Subalgebra.LinearDisjoint.exists_field_of_isDomain_of_injective F A B
      (RingHom.injective _) (RingHom.injective _)
  ⟨K, inst1, inst2, fa, fb, linearDisjoint_iff'.2 H⟩


variable (F) in
/-- If for any field extension `K` of `F` that `A` and `B` embed into, their images are
linearly disjoint, then `A ⊗[F] B` is a field. (In the proof we choose `K` to be the quotient
of `A ⊗[F] B` by a maximal ideal.) -/
theorem isField_of_forall (A : Type v) [Field A] (B : Type w) [Field B]
    [Algebra F A] [Algebra F B]
    (H : ∀ (K : Type (max v w)) [Field K] [Algebra F K],
      ∀ (fa : A →ₐ[F] K) (fb : B →ₐ[F] K), fa.fieldRange.LinearDisjoint fb.fieldRange) :
    IsField (A ⊗[F] B) := by
  /-
    F : Type u
    inst✝⁴ : Field F
    A : Type v
    inst✝³ : Field A
    B : Type w
    inst✝² : Field B
    inst✝¹ : Algebra F A
    inst✝ : Algebra F B
    H : ∀ (K : Type (max v w)) [inst : Field K] [inst_1 : Algebra F K] (fa : AlgHo …
    ⊢ IsField (TensorProduct F A B)
  -/
  obtain ⟨M, hM⟩ := Ideal.exists_maximal (A ⊗[F] B)
  /-
    case intro
    F : Type u
    inst✝⁴ : Field F
    A : Type v
    inst✝³ : Field A
    B : Type w
    inst✝² : Field B
    inst✝¹ : Algebra F A
    inst✝ : Algebra F B
    H : ∀ (K : Type (max v w)) [inst : Field K] [inst_1 : Algebra F K] (fa : AlgHo …
    M : Ideal (TensorProduct F A B)
    hM : M.IsMaximal
    ⊢ IsField (TensorProduct F A B)
  -/
  apply not_imp_not.1 (Ring.ne_bot_of_isMaximal_of_not_isField hM)
  /-
    case intro
    F : Type u
    inst✝⁴ : Field F
    A : Type v
    inst✝³ : Field A
    B : Type w
    inst✝² : Field B
    inst✝¹ : Algebra F A
    inst✝ : Algebra F B
    H : ∀ (K : Type (max v w)) [inst : Field K] [inst_1 : Algebra F K] (fa : AlgHo …
    M : Ideal (TensorProduct F A B)
    hM : M.IsMaximal
    ⊢ Eq M Bot.bot
  -/
  let K : Type (max v w) := A ⊗[F] B ⧸ M
  /-
    case intro
    F : Type u
    inst✝⁴ : Field F
    A : Type v
    inst✝³ : Field A
    B : Type w
    inst✝² : Field B
    inst✝¹ : Algebra F A
    inst✝ : Algebra F B
    H : ∀ (K : Type (max v w)) [inst : Field K] [inst_1 : Algebra F K] (fa : AlgHo …
    M : Ideal (TensorProduct F A B)
    hM : M.IsMaximal
    K : Type (max v w) := HasQuotient.Quotient (TensorProduct F A B) M
    ⊢ Eq M Bot.bot
  -/
  letI : Field K := Ideal.Quotient.field _
  /-
    case intro
    F : Type u
    inst✝⁴ : Field F
    A : Type v
    inst✝³ : Field A
    B : Type w
    inst✝² : Field B
    inst✝¹ : Algebra F A
    inst✝ : Algebra F B
    H : ∀ (K : Type (max v w)) [inst : Field K] [inst_1 : Algebra F K] (fa : AlgHo …
    M : Ideal (TensorProduct F A B)
    hM : M.IsMaximal
    K : Type (max v w) := HasQuotient.Quotient (TensorProduct F A B) M
    this : Field K := Ideal.Quotient.field M
    ⊢ Eq M Bot.bot
  -/
  let i := IsScalarTower.toAlgHom F (A ⊗[F] B) K
  /-
    case intro
    F : Type u
    inst✝⁴ : Field F
    A : Type v
    inst✝³ : Field A
    B : Type w
    inst✝² : Field B
    inst✝¹ : Algebra F A
    inst✝ : Algebra F B
    H : ∀ (K : Type (max v w)) [inst : Field K] [inst_1 : Algebra F K] (fa : AlgHo …
    M : Ideal (TensorProduct F A B)
    hM : M.IsMaximal
    K : Type (max v w) := HasQuotient.Quotient (TensorProduct F A B) M
    this : Field K := Ideal.Quotient.field M
    i : AlgHom F (TensorProduct F A B) K := IsScalarTower.toAlgHom F (TensorProduc …
    ⊢ Eq M Bot.bot
  -/
  let fa := i.comp (Algebra.TensorProduct.includeLeft : A →ₐ[F] _)
  /-
    case intro
    F : Type u
    inst✝⁴ : Field F
    A : Type v
    inst✝³ : Field A
    B : Type w
    inst✝² : Field B
    inst✝¹ : Algebra F A
    inst✝ : Algebra F B
    H : ∀ (K : Type (max v w)) [inst : Field K] [inst_1 : Algebra F K] (fa : AlgHo …
    M : Ideal (TensorProduct F A B)
    hM : M.IsMaximal
    K : Type (max v w) := HasQuotient.Quotient (TensorProduct F A B) M
    this : Field K := Ideal.Quotient.field M
    i : AlgHom F (TensorProduct F A B) K := IsScalarTower.toAlgHom F (TensorProduc …
    fa : AlgHom F A K := i.comp Algebra.TensorProduct.includeLeft
    ⊢ Eq M Bot.bot
  -/
  let fb := i.comp (Algebra.TensorProduct.includeRight : B →ₐ[F] _)
  /-
    case intro
    F : Type u
    inst✝⁴ : Field F
    A : Type v
    inst✝³ : Field A
    B : Type w
    inst✝² : Field B
    inst✝¹ : Algebra F A
    inst✝ : Algebra F B
    H : ∀ (K : Type (max v w)) [inst : Field K] [inst_1 : Algebra F K] (fa : AlgHo …
    M : Ideal (TensorProduct F A B)
    hM : M.IsMaximal
    K : Type (max v w) := HasQuotient.Quotient (TensorProduct F A B) M
    this : Field K := Ideal.Quotient.field M
    i : AlgHom F (TensorProduct F A B) K := IsScalarTower.toAlgHom F (TensorProduc …
    fa : AlgHom F A K := i.comp Algebra.TensorProduct.includeLeft
    fb : AlgHom F B K := i.comp Algebra.TensorProduct.includeRight
    ⊢ Eq M Bot.bot
  -/
  replace H := H K fa fb
  simp_rw [linearDisjoint_iff', AlgHom.fieldRange_toSubalgebra,
    Subalgebra.linearDisjoint_iff_injective] at H
  have hi : i = (fa.range.mulMap fb.range).comp (Algebra.TensorProduct.congr
      (AlgEquiv.ofInjective fa fa.injective) (AlgEquiv.ofInjective fb fb.injective)) := by
    ext <;> simp [fa, fb]
  /-
    case intro
    F : Type u
    inst✝⁴ : Field F
    A : Type v
    inst✝³ : Field A
    B : Type w
    inst✝² : Field B
    inst✝¹ : Algebra F A
    inst✝ : Algebra F B
    M : Ideal (TensorProduct F A B)
    hM : M.IsMaximal
    K : Type (max v w) := HasQuotient.Quotient (TensorProduct F A B) M
    this : Field K := Ideal.Quotient.field M
    i : AlgHom F (TensorProduct F A B) K := IsScalarTower.toAlgHom F (TensorProduc …
    fa : AlgHom F A K := i.comp Algebra.TensorProduct.includeLeft
    fb : AlgHom F B K := i.comp Algebra.TensorProduct.includeRight
    H : Function.Injective ⇑(fa.range.mulMap fb.range)
    hi : Eq i ((fa.range.mulMap fb.range).comp ↑(Algebra.TensorProduct.congr (AlgE …
    ⊢ Eq M Bot.bot
  -/
  replace H : Function.Injective i := by simpa [hi]
  /-
    case intro
    F : Type u
    inst✝⁴ : Field F
    A : Type v
    inst✝³ : Field A
    B : Type w
    inst✝² : Field B
    inst✝¹ : Algebra F A
    inst✝ : Algebra F B
    M : Ideal (TensorProduct F A B)
    hM : M.IsMaximal
    K : Type (max v w) := HasQuotient.Quotient (TensorProduct F A B) M
    this : Field K := Ideal.Quotient.field M
    i : AlgHom F (TensorProduct F A B) K := IsScalarTower.toAlgHom F (TensorProduc …
    fa : AlgHom F A K := i.comp Algebra.TensorProduct.includeLeft
    fb : AlgHom F B K := i.comp Algebra.TensorProduct.includeRight
    hi : Eq i ((fa.range.mulMap fb.range).comp ↑(Algebra.TensorProduct.congr (AlgE …
    H : Function.Injective ⇑i
    ⊢ Eq M Bot.bot
  -/
  change Function.Injective (Ideal.Quotient.mk M) at H
  /-
    case intro
    F : Type u
    inst✝⁴ : Field F
    A : Type v
    inst✝³ : Field A
    B : Type w
    inst✝² : Field B
    inst✝¹ : Algebra F A
    inst✝ : Algebra F B
    M : Ideal (TensorProduct F A B)
    hM : M.IsMaximal
    K : Type (max v w) := HasQuotient.Quotient (TensorProduct F A B) M
    this : Field K := Ideal.Quotient.field M
    i : AlgHom F (TensorProduct F A B) K := IsScalarTower.toAlgHom F (TensorProduc …
    fa : AlgHom F A K := i.comp Algebra.TensorProduct.includeLeft
    fb : AlgHom F B K := i.comp Algebra.TensorProduct.includeRight
    hi : Eq i ((fa.range.mulMap fb.range).comp ↑(Algebra.TensorProduct.congr (AlgE …
    H : Function.Injective ⇑(Ideal.Quotient.mk M)
    ⊢ Eq M Bot.bot
  -/
  rwa [RingHom.injective_iff_ker_eq_bot, Ideal.mk_ker] at H
  /-
    🎉 no goals
  -/


variable (F E) in
/-- If `E` and `K` are field extensions of `F`, one of them is algebraic, such that
`E ⊗[F] K` is a domain, then `E ⊗[F] K` is also a field. It is a corollary of
`Subalgebra.LinearDisjoint.exists_field_of_isDomain_of_injective` and
`IntermediateField.sup_toSubalgebra_of_isAlgebraic`.
See `Algebra.TensorProduct.isAlgebraic_of_isField` for its converse (in an earlier file). -/
theorem _root_.Algebra.TensorProduct.isField_of_isAlgebraic
    (K : Type*) [Field K] [Algebra F K] [IsDomain (E ⊗[F] K)]
    (halg : Algebra.IsAlgebraic F E ∨ Algebra.IsAlgebraic F K) : IsField (E ⊗[F] K) :=
  have ⟨L, _, _, fa, fb, hfa, hfb, H⟩ :=
    Subalgebra.LinearDisjoint.exists_field_of_isDomain_of_injective F E K
      (RingHom.injective _) (RingHom.injective _)
  let f : E ⊗[F] K ≃ₐ[F] ↥(fa.fieldRange ⊔ fb.fieldRange) :=
    Algebra.TensorProduct.congr (AlgEquiv.ofInjective fa hfa) (AlgEquiv.ofInjective fb hfb)
    |>.trans (Subalgebra.LinearDisjoint.mulMap H)
    |>.trans (Subalgebra.equivOfEq _ _
      (sup_toSubalgebra_of_isAlgebraic fa.fieldRange fb.fieldRange <| by
        rwa [(AlgEquiv.ofInjective fa hfa).isAlgebraic_iff,
          (AlgEquiv.ofInjective fb hfb).isAlgebraic_iff] at halg).symm)
  f.toMulEquiv.isField _ (Field.toIsField _)


/-- If `A` and `L` are linearly disjoint over `F` and one of them is algebraic,
then `A ⊗[F] L` is a field. -/
theorem isField_of_isAlgebraic (H : A.LinearDisjoint L)
    (halg : Algebra.IsAlgebraic F A ∨ Algebra.IsAlgebraic F L) : IsField (A ⊗[F] L) :=
  have := H.isDomain
  Algebra.TensorProduct.isField_of_isAlgebraic F A L halg


/-- If `A` and `B` are field extensions of `F`, one of them is algebraic, such that there exists a
field `E` that `A` and `B` embeds into with linearly disjoint images, then `A ⊗[F] B`
is a field. -/
theorem isField_of_isAlgebraic' {A B : Type*} [Field A] [Algebra F A] [Field B] [Algebra F B]
    {fa : A →ₐ[F] E} {fb : B →ₐ[F] E} (H : fa.fieldRange.LinearDisjoint fb.fieldRange)
    (halg : Algebra.IsAlgebraic F A ∨ Algebra.IsAlgebraic F B) : IsField (A ⊗[F] B) :=
  have := H.isDomain'
  Algebra.TensorProduct.isField_of_isAlgebraic F A B halg


/-- If `A` and `L` are linearly disjoint, one of them is algebraic, then for any `B` and `L'`
isomorphic to `A` and `L` respectively, `B` and `L'` are also linearly disjoint. -/
theorem algEquiv_of_isAlgebraic (H : A.LinearDisjoint L)
    {E' : Type*} [Field E'] [Algebra F E']
    (B : IntermediateField F E')
    (L' : Type*) [Field L'] [Algebra F L'] [Algebra L' E'] [IsScalarTower F L' E']
    (f1 : A ≃ₐ[F] B) (f2 : L ≃ₐ[F] L')
    (halg : Algebra.IsAlgebraic F A ∨ Algebra.IsAlgebraic F L) :
    B.LinearDisjoint L' :=
  .of_isField ((Algebra.TensorProduct.congr f1 f2).symm.toMulEquiv.isField _
    (H.isField_of_isAlgebraic halg))


