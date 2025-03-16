/-- A linearly independent family of nonzero vectors gives an independent family of points
in projective space. -/
inductive Independent : (ι → ℙ K V) → Prop
  | mk (f : ι → V) (hf : ∀ i : ι, f i ≠ 0) (hl : LinearIndependent K f) :
    Independent fun i => mk K (f i) (hf i)


/-- A family of points in a projective space is independent if and only if the representative
vectors determined by the family are linearly independent. -/
theorem independent_iff : Independent f ↔ LinearIndependent K (Projectivization.rep ∘ f) := by
  /-
    ι : Type u_1
    K : Type u_2
    V : Type u_3
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : ι → Projectivization K V
    ⊢ Iff (Projectivization.Independent f) (LinearIndependent K (Function.comp Pro …
  -/
  refine ⟨?_, fun h => ?_⟩
    /-
      case refine_1
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : ι → Projectivization K V
      ⊢ Projectivization.Independent f → LinearIndependent K (Function.comp Projecti …
    -/
  · rintro ⟨ff, hff, hh⟩
    /-
      case refine_1.mk
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      ff : ι → V
      hff : ∀ (i : ι), Ne (ff i) 0
      hh : LinearIndependent K ff
      ⊢ LinearIndependent K (Function.comp Projectivization.rep fun i => Projectiviz …
    -/
    choose a ha using fun i : ι => exists_smul_eq_mk_rep K (ff i) (hff i)
    /-
      case refine_1.mk
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      ff : ι → V
      hff : ∀ (i : ι), Ne (ff i) 0
      hh : LinearIndependent K ff
      a : ι → Units K
      ha : ∀ (i : ι), Eq (HSMul.hSMul (a i) (ff i)) (Projectivization.mk K (ff i) ⋯) …
      ⊢ LinearIndependent K (Function.comp Projectivization.rep fun i => Projectiviz …
    -/
    convert hh.units_smul a
    /-
      case h.e'_4
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      ff : ι → V
      hff : ∀ (i : ι), Ne (ff i) 0
      hh : LinearIndependent K ff
      a : ι → Units K
      ha : ∀ (i : ι), Eq (HSMul.hSMul (a i) (ff i)) (Projectivization.mk K (ff i) ⋯) …
      ⊢ Eq (Function.comp Projectivization.rep fun i => Projectivization.mk K (ff i) …
    -/
    ext i
    /-
      case h.e'_4.h
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      ff : ι → V
      hff : ∀ (i : ι), Ne (ff i) 0
      hh : LinearIndependent K ff
      a : ι → Units K
      ha : ∀ (i : ι), Eq (HSMul.hSMul (a i) (ff i)) (Projectivization.mk K (ff i) ⋯) …
      i : ι
      ⊢ Eq (Function.comp Projectivization.rep (fun i => Projectivization.mk K (ff i …
    -/
    exact (ha i).symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : ι → Projectivization K V
      h : LinearIndependent K (Function.comp Projectivization.rep f)
      ⊢ Projectivization.Independent f
    -/
  · convert Independent.mk _ _ h
      /-
        case h.e'_7.h
        ι : Type u_1
        K : Type u_2
        V : Type u_3
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        f : ι → Projectivization K V
        h : LinearIndependent K (Function.comp Projectivization.rep f)
        x✝ : ι
        ⊢ Eq (f x✝) (Projectivization.mk K (Function.comp Projectivization.rep f x✝) ⋯)
      -/
    · simp only [mk_rep, Function.comp_apply]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        ι : Type u_1
        K : Type u_2
        V : Type u_3
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        f : ι → Projectivization K V
        h : LinearIndependent K (Function.comp Projectivization.rep f)
        ⊢ ∀ (i : ι), Ne (Function.comp Projectivization.rep f i) 0
      -/
    · intro i
      /-
        case refine_2
        ι : Type u_1
        K : Type u_2
        V : Type u_3
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        f : ι → Projectivization K V
        h : LinearIndependent K (Function.comp Projectivization.rep f)
        i : ι
        ⊢ Ne (Function.comp Projectivization.rep f i) 0
      -/
      apply rep_nonzero
      /-
        🎉 no goals
      -/


/-- A family of points in projective space is independent if and only if the family of
submodules which the points determine is independent in the lattice-theoretic sense. -/
theorem independent_iff_iSupIndep : Independent f ↔ iSupIndep fun i => (f i).submodule := by
  /-
    ι : Type u_1
    K : Type u_2
    V : Type u_3
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : ι → Projectivization K V
    ⊢ Iff (Projectivization.Independent f) (iSupIndep fun i => (f i).submodule)
  -/
  refine ⟨?_, fun h => ?_⟩
    /-
      case refine_1
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : ι → Projectivization K V
      ⊢ Projectivization.Independent f → iSupIndep fun i => (f i).submodule
    -/
  · rintro ⟨f, hf, hi⟩
    /-
      case refine_1.mk
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : ι → V
      hf : ∀ (i : ι), Ne (f i) 0
      hi : LinearIndependent K f
      ⊢ iSupIndep fun i => ((fun i => Projectivization.mk K (f i) ⋯) i).submodule
    -/
    simp only [submodule_mk]
    /-
      case refine_1.mk
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : ι → V
      hf : ∀ (i : ι), Ne (f i) 0
      hi : LinearIndependent K f
      ⊢ iSupIndep fun i => Submodule.span K (Singleton.singleton (f i))
    -/
    exact (iSupIndep_iff_linearIndependent_of_ne_zero (R := K) hf).mpr hi
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : ι → Projectivization K V
      h : iSupIndep fun i => (f i).submodule
      ⊢ Projectivization.Independent f
    -/
  · rw [independent_iff]
    /-
      case refine_2
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : ι → Projectivization K V
      h : iSupIndep fun i => (f i).submodule
      ⊢ LinearIndependent K (Function.comp Projectivization.rep f)
    -/
    refine h.linearIndependent (Projectivization.submodule ∘ f) (fun i => ?_) fun i => ?_
      /-
        case refine_2.refine_1
        ι : Type u_1
        K : Type u_2
        V : Type u_3
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        f : ι → Projectivization K V
        h : iSupIndep fun i => (f i).submodule
        i : ι
        ⊢ Membership.mem (Function.comp Projectivization.submodule f i) (Function.comp …
      -/
    · simpa only [Function.comp_apply, submodule_eq] using Submodule.mem_span_singleton_self _
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        ι : Type u_1
        K : Type u_2
        V : Type u_3
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        f : ι → Projectivization K V
        h : iSupIndep fun i => (f i).submodule
        i : ι
        ⊢ Ne (Function.comp Projectivization.rep f i) 0
      -/
    · exact rep_nonzero (f i)
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-11-24")]
alias independent_iff_completeLattice_independent := independent_iff_iSupIndep


/-- A linearly dependent family of nonzero vectors gives a dependent family of points
in projective space. -/
inductive Dependent : (ι → ℙ K V) → Prop
  | mk (f : ι → V) (hf : ∀ i : ι, f i ≠ 0) (h : ¬LinearIndependent K f) :
    Dependent fun i => mk K (f i) (hf i)


/-- A family of points in a projective space is dependent if and only if their
representatives are linearly dependent. -/
theorem dependent_iff : Dependent f ↔ ¬LinearIndependent K (Projectivization.rep ∘ f) := by
  /-
    ι : Type u_1
    K : Type u_2
    V : Type u_3
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : ι → Projectivization K V
    ⊢ Iff (Projectivization.Dependent f) (Not (LinearIndependent K (Function.comp  …
  -/
  refine ⟨?_, fun h => ?_⟩
    /-
      case refine_1
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : ι → Projectivization K V
      ⊢ Projectivization.Dependent f → Not (LinearIndependent K (Function.comp Proje …
    -/
  · rintro ⟨ff, hff, hh1⟩
    /-
      case refine_1.mk
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      ff : ι → V
      hff : ∀ (i : ι), Ne (ff i) 0
      hh1 : Not (LinearIndependent K ff)
      ⊢ Not (LinearIndependent K (Function.comp Projectivization.rep fun i => Projec …
    -/
    contrapose! hh1
    /-
      case refine_1.mk
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      ff : ι → V
      hff : ∀ (i : ι), Ne (ff i) 0
      hh1 : LinearIndependent K (Function.comp Projectivization.rep fun i => Project …
      ⊢ LinearIndependent K ff
    -/
    choose a ha using fun i : ι => exists_smul_eq_mk_rep K (ff i) (hff i)
    /-
      case refine_1.mk
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      ff : ι → V
      hff : ∀ (i : ι), Ne (ff i) 0
      hh1 : LinearIndependent K (Function.comp Projectivization.rep fun i => Project …
      a : ι → Units K
      ha : ∀ (i : ι), Eq (HSMul.hSMul (a i) (ff i)) (Projectivization.mk K (ff i) ⋯) …
      ⊢ LinearIndependent K ff
    -/
    convert hh1.units_smul a⁻¹
    /-
      case h.e'_4
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      ff : ι → V
      hff : ∀ (i : ι), Ne (ff i) 0
      hh1 : LinearIndependent K (Function.comp Projectivization.rep fun i => Project …
      a : ι → Units K
      ha : ∀ (i : ι), Eq (HSMul.hSMul (a i) (ff i)) (Projectivization.mk K (ff i) ⋯) …
      ⊢ Eq ff (HSMul.hSMul (Inv.inv a) (Function.comp Projectivization.rep fun i =>  …
    -/
    ext i
    /-
      case h.e'_4.h
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      ff : ι → V
      hff : ∀ (i : ι), Ne (ff i) 0
      hh1 : LinearIndependent K (Function.comp Projectivization.rep fun i => Project …
      a : ι → Units K
      ha : ∀ (i : ι), Eq (HSMul.hSMul (a i) (ff i)) (Projectivization.mk K (ff i) ⋯) …
      i : ι
      ⊢ Eq (ff i) (HSMul.hSMul (Inv.inv a) (Function.comp Projectivization.rep fun i …
    -/
    simp only [← ha, inv_smul_smul, Pi.smul_apply', Pi.inv_apply, Function.comp_apply]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      K : Type u_2
      V : Type u_3
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : ι → Projectivization K V
      h : Not (LinearIndependent K (Function.comp Projectivization.rep f))
      ⊢ Projectivization.Dependent f
    -/
  · convert Dependent.mk _ _ h
      /-
        case h.e'_7.h
        ι : Type u_1
        K : Type u_2
        V : Type u_3
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        f : ι → Projectivization K V
        h : Not (LinearIndependent K (Function.comp Projectivization.rep f))
        x✝ : ι
        ⊢ Eq (f x✝) (Projectivization.mk K (Function.comp Projectivization.rep f x✝) ⋯)
      -/
    · simp only [mk_rep, Function.comp_apply]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        ι : Type u_1
        K : Type u_2
        V : Type u_3
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        f : ι → Projectivization K V
        h : Not (LinearIndependent K (Function.comp Projectivization.rep f))
        ⊢ ∀ (i : ι), Ne (Function.comp Projectivization.rep f i) 0
      -/
    · exact fun i => rep_nonzero (f i)
      /-
        🎉 no goals
      -/


/-- Dependence is the negation of independence. -/
theorem dependent_iff_not_independent : Dependent f ↔ ¬Independent f := by
  /-
    ι : Type u_1
    K : Type u_2
    V : Type u_3
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : ι → Projectivization K V
    ⊢ Iff (Projectivization.Dependent f) (Not (Projectivization.Independent f))
  -/
  rw [dependent_iff, independent_iff]
  /-
    🎉 no goals
  -/


/-- Independence is the negation of dependence. -/
theorem independent_iff_not_dependent : Independent f ↔ ¬Dependent f := by
  /-
    ι : Type u_1
    K : Type u_2
    V : Type u_3
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : ι → Projectivization K V
    ⊢ Iff (Projectivization.Independent f) (Not (Projectivization.Dependent f))
  -/
  rw [dependent_iff_not_independent, Classical.not_not]
  /-
    🎉 no goals
  -/


/-- Two points in a projective space are dependent if and only if they are equal. -/
@[simp]
theorem dependent_pair_iff_eq (u v : ℙ K V) : Dependent ![u, v] ↔ u = v := by
  rw [dependent_iff_not_independent, independent_iff, linearIndependent_fin2,
    Function.comp_apply, Matrix.cons_val_one, Matrix.head_cons, Ne]
  simp only [Matrix.cons_val_zero, not_and, not_forall, Classical.not_not, Function.comp_apply,
    ← mk_eq_mk_iff' K _ _ (rep_nonzero u) (rep_nonzero v), mk_rep, Classical.imp_iff_right_iff]
  /-
    K : Type u_2
    V : Type u_3
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    u v : Projectivization K V
    ⊢ Or (Not (Eq v.rep 0)) (Eq u v)
  -/
  exact Or.inl (rep_nonzero v)
  /-
    🎉 no goals
  -/


/-- Two points in a projective space are independent if and only if the points are not equal. -/
@[simp]
theorem independent_pair_iff_neq (u v : ℙ K V) : Independent ![u, v] ↔ u ≠ v := by
  /-
    K : Type u_2
    V : Type u_3
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    u v : Projectivization K V
    ⊢ Iff (Projectivization.Independent (Matrix.vecCons u (Matrix.vecCons v Matrix …
  -/
  rw [independent_iff_not_dependent, dependent_pair_iff_eq u v]
  /-
    🎉 no goals
  -/


