/-- If every finite set of linearly independent vectors has cardinality at most `n`,
then the same is true for arbitrary sets of linearly independent vectors.
-/
theorem linearIndependent_bounded_of_finset_linearIndependent_bounded {n : ℕ}
    (H : ∀ s : Finset M, (LinearIndependent R fun i : s => (i : M)) → s.card ≤ n) :
    ∀ s : Set M, LinearIndependent R ((↑) : s → M) → #s ≤ n := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    H : ∀ (s : Finset M), (LinearIndependent R fun i => ↑i) → LE.le s.card n
    ⊢ ∀ (s : Set M), LinearIndependent R Subtype.val → LE.le (Cardinal.mk ↑s) ↑n
  -/
  intro s li
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    H : ∀ (s : Finset M), (LinearIndependent R fun i => ↑i) → LE.le s.card n
    s : Set M
    li : LinearIndependent R Subtype.val
    ⊢ LE.le (Cardinal.mk ↑s) ↑n
  -/
  apply Cardinal.card_le_of
  /-
    case H
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    H : ∀ (s : Finset M), (LinearIndependent R fun i => ↑i) → LE.le s.card n
    s : Set M
    li : LinearIndependent R Subtype.val
    ⊢ ∀ (s_1 : Finset ↑s), LE.le s_1.card n
  -/
  intro t
  /-
    case H
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    H : ∀ (s : Finset M), (LinearIndependent R fun i => ↑i) → LE.le s.card n
    s : Set M
    li : LinearIndependent R Subtype.val
    t : Finset ↑s
    ⊢ LE.le t.card n
  -/
  rw [← Finset.card_map (Embedding.subtype s)]
  /-
    case H
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    H : ∀ (s : Finset M), (LinearIndependent R fun i => ↑i) → LE.le s.card n
    s : Set M
    li : LinearIndependent R Subtype.val
    t : Finset ↑s
    ⊢ LE.le (Finset.map (Function.Embedding.subtype s) t).card n
  -/
  apply H
  /-
    case H.a
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    H : ∀ (s : Finset M), (LinearIndependent R fun i => ↑i) → LE.le s.card n
    s : Set M
    li : LinearIndependent R Subtype.val
    t : Finset ↑s
    ⊢ LinearIndependent R fun i => ↑i
  -/
  apply linearIndependent_finset_map_embedding_subtype _ li
  /-
    🎉 no goals
  -/


theorem rank_le {n : ℕ}
    (H : ∀ s : Finset M, (LinearIndependent R fun i : s => (i : M)) → s.card ≤ n) :
    Module.rank R M ≤ n := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    H : ∀ (s : Finset M), (LinearIndependent R fun i => ↑i) → LE.le s.card n
    ⊢ LE.le (Module.rank R M) ↑n
  -/
  rw [Module.rank_def]
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    H : ∀ (s : Finset M), (LinearIndependent R fun i => ↑i) → LE.le s.card n
    ⊢ LE.le (iSup fun ι => Cardinal.mk ↑↑ι) ↑n
  -/
  apply ciSup_le'
  /-
    case h
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    H : ∀ (s : Finset M), (LinearIndependent R fun i => ↑i) → LE.le s.card n
    ⊢ ∀ (i : Subtype fun s => LinearIndependent R Subtype.val), LE.le (Cardinal.mk …
  -/
  rintro ⟨s, li⟩
  /-
    case h.mk
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    H : ∀ (s : Finset M), (LinearIndependent R fun i => ↑i) → LE.le s.card n
    s : Set M
    li : LinearIndependent R Subtype.val
    ⊢ LE.le (Cardinal.mk ↑↑⟨s, li⟩) ↑n
  -/
  exact linearIndependent_bounded_of_finset_linearIndependent_bounded H _ li
  /-
    🎉 no goals
  -/


/-- See `rank_zero_iff` for a stronger version with `NoZeroSMulDivisor R M`. -/
lemma rank_eq_zero_iff :
    Module.rank R M = 0 ↔ ∀ x : M, ∃ a : R, a ≠ 0 ∧ a • x = 0 := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Eq (Module.rank R M) 0) (∀ (x : M), Exists fun a => And (Ne a 0) (Eq (H …
  -/
  nontriviality R
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    a✝ : Nontrivial R
    ⊢ Iff (Eq (Module.rank R M) 0) (∀ (x : M), Exists fun a => And (Ne a 0) (Eq (H …
  -/
  constructor
    /-
      case mp
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      ⊢ Eq (Module.rank R M) 0 → ∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul. …
    -/
  · contrapose!
    /-
      case mp
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      ⊢ (Exists fun x => ∀ (a : R), Ne a 0 → Ne (HSMul.hSMul a x) 0) → Ne (Module.ra …
    -/
    rintro ⟨x, hx⟩
    /-
      case mp.intro
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      x : M
      hx : ∀ (a : R), Ne a 0 → Ne (HSMul.hSMul a x) 0
      ⊢ Ne (Module.rank R M) 0
    -/
    rw [← Cardinal.one_le_iff_ne_zero]
    have : LinearIndependent R (fun _ : Unit ↦ x) :=
      linearIndependent_iff.mpr (fun l hl ↦ Finsupp.unique_ext <| not_not.mp fun H ↦
        hx _ H ((Finsupp.linearCombination_unique _ _ _).symm.trans hl))
    /-
      case mp.intro
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      x : M
      hx : ∀ (a : R), Ne a 0 → Ne (HSMul.hSMul a x) 0
      this : LinearIndependent R fun x_1 => x
      ⊢ LE.le 1 (Module.rank R M)
    -/
    simpa using this.cardinal_lift_le_rank
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      ⊢ (∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a x) 0)) → Eq (Mod …
    -/
  · intro h
    /-
      case mpr
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      h : ∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a x) 0)
      ⊢ Eq (Module.rank R M) 0
    -/
    rw [← le_zero_iff, Module.rank_def]
    /-
      case mpr
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      h : ∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a x) 0)
      ⊢ LE.le (iSup fun ι => Cardinal.mk ↑↑ι) 0
    -/
    apply ciSup_le'
    /-
      case mpr.h
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      h : ∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a x) 0)
      ⊢ ∀ (i : Subtype fun s => LinearIndependent R Subtype.val), LE.le (Cardinal.mk …
    -/
    intro ⟨s, hs⟩
    /-
      case mpr.h
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      h : ∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a x) 0)
      s : Set M
      hs : LinearIndependent R Subtype.val
      ⊢ LE.le (Cardinal.mk ↑↑⟨s, hs⟩) 0
    -/
    rw [nonpos_iff_eq_zero, Cardinal.mk_eq_zero_iff, ← not_nonempty_iff]
    /-
      case mpr.h
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      h : ∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a x) 0)
      s : Set M
      hs : LinearIndependent R Subtype.val
      ⊢ Not (Nonempty ↑↑⟨s, hs⟩)
    -/
    rintro ⟨i : s⟩
    /-
      case mpr.h.intro
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      h : ∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a x) 0)
      s : Set M
      hs : LinearIndependent R Subtype.val
      i : ↑s
      ⊢ False
    -/
    obtain ⟨a, ha, ha'⟩ := h i
    /-
      case mpr.h.intro.intro.intro
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      h : ∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a x) 0)
      s : Set M
      hs : LinearIndependent R Subtype.val
      i : ↑s
      a : R
      ha : Ne a 0
      ha' : Eq (HSMul.hSMul a ↑i) 0
      ⊢ False
    -/
    apply ha
    /-
      case mpr.h.intro.intro.intro
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      a✝ : Nontrivial R
      h : ∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a x) 0)
      s : Set M
      hs : LinearIndependent R Subtype.val
      i : ↑s
      a : R
      ha : Ne a 0
      ha' : Eq (HSMul.hSMul a ↑i) 0
      ⊢ Eq a 0
    -/
    simpa using DFunLike.congr_fun (linearIndependent_iff.mp hs (Finsupp.single i a) (by simpa)) i
    /-
      🎉 no goals
    -/


theorem rank_zero_iff_forall_zero :
    Module.rank R M = 0 ↔ ∀ x : M, x = 0 := by
  simp_rw [rank_eq_zero_iff, smul_eq_zero, and_or_left, not_and_self_iff, false_or,
    exists_and_right, and_iff_right (exists_ne (0 : R))]


/-- See `rank_subsingleton` for the reason that `Nontrivial R` is needed.
Also see `rank_eq_zero_iff` for the version without `NoZeroSMulDivisor R M`. -/
theorem rank_zero_iff : Module.rank R M = 0 ↔ Subsingleton M :=
  rank_zero_iff_forall_zero.trans (subsingleton_iff_forall_eq 0).symm


theorem rank_pos_iff_exists_ne_zero : 0 < Module.rank R M ↔ ∃ x : M, x ≠ 0 := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroSMulDivisors R M
    ⊢ Iff (LT.lt 0 (Module.rank R M)) (Exists fun x => Ne x 0)
  -/
  rw [← not_iff_not]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroSMulDivisors R M
    ⊢ Iff (Not (LT.lt 0 (Module.rank R M))) (Not (Exists fun x => Ne x 0))
  -/
  simpa using rank_zero_iff_forall_zero
  /-
    🎉 no goals
  -/


theorem rank_pos_iff_nontrivial : 0 < Module.rank R M ↔ Nontrivial M :=
  rank_pos_iff_exists_ne_zero.trans (nontrivial_iff_exists_ne 0).symm


theorem rank_pos [Nontrivial M] : 0 < Module.rank R M :=
  rank_pos_iff_nontrivial.mpr ‹_›


/-- See `rank_subsingleton` that assumes `Subsingleton R` instead. -/
theorem rank_subsingleton' [Subsingleton M] : Module.rank R M = 0 :=
  rank_eq_zero_iff.mpr fun _ ↦ ⟨1, one_ne_zero, Subsingleton.elim _ _⟩


@[simp]
theorem rank_punit : Module.rank R PUnit = 0 := rank_subsingleton' _ _


@[simp]
theorem rank_bot : Module.rank R (⊥ : Submodule R M) = 0 := rank_subsingleton' _ _


theorem exists_mem_ne_zero_of_rank_pos {s : Submodule R M} (h : 0 < Module.rank R s) :
    ∃ b : M, b ∈ s ∧ b ≠ 0 :=
                                            /-
                                              R : Type u
                                              M : Type v
                                              inst✝³ : Ring R
                                              inst✝² : AddCommGroup M
                                              inst✝¹ : Module R M
                                              inst✝ : Nontrivial R
                                              s : Submodule R M
                                              h : LT.lt 0 (Module.rank R (Subtype fun x => Membership.mem s x))
                                              eq : Eq s Bot.bot
                                              ⊢ False
                                            -/
  exists_mem_ne_zero_of_ne_bot fun eq => by rw [eq, rank_bot] at h; exact lt_irrefl _ h
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem Module.finite_of_rank_eq_nat [Module.Free R M] {n : ℕ} (h : Module.rank R M = n) :
    Module.Finite R M := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    n : Nat
    h : Eq (Module.rank R M) ↑n
    ⊢ Module.Finite R M
  -/
  nontriviality R
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    n : Nat
    h : Eq (Module.rank R M) ↑n
    a✝ : Nontrivial R
    ⊢ Module.Finite R M
  -/
  obtain ⟨⟨ι, b⟩⟩ := Module.Free.exists_basis (R := R) (M := M)
  have := mk_lt_aleph0_iff.mp <|
    b.linearIndependent.cardinal_le_rank |>.trans_eq h |>.trans_lt <| nat_lt_aleph0 n
  /-
    case intro.mk
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    n : Nat
    h : Eq (Module.rank R M) ↑n
    a✝ : Nontrivial R
    ι : Type v
    b : Basis ι R M
    this : Finite ι
    ⊢ Module.Finite R M
  -/
  exact Module.Finite.of_basis b
  /-
    🎉 no goals
  -/


theorem Module.finite_of_rank_eq_zero [NoZeroSMulDivisors R M]
    (h : Module.rank R M = 0) :
    Module.Finite R M := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    h : Eq (Module.rank R M) 0
    ⊢ Module.Finite R M
  -/
  nontriviality R
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    h : Eq (Module.rank R M) 0
    a✝ : Nontrivial R
    ⊢ Module.Finite R M
  -/
  rw [rank_zero_iff] at h
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    h : Subsingleton M
    a✝ : Nontrivial R
    ⊢ Module.Finite R M
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem Module.finite_of_rank_eq_one [Module.Free R M] (h : Module.rank R M = 1) :
    Module.Finite R M :=
  Module.finite_of_rank_eq_nat <| h.trans Nat.cast_one.symm


/-- If a module has a finite dimension, all bases are indexed by a finite type. -/
theorem Basis.nonempty_fintype_index_of_rank_lt_aleph0 {ι : Type*} (b : Basis ι R M)
    (h : Module.rank R M < ℵ₀) : Nonempty (Fintype ι) := by
  rwa [← Cardinal.lift_lt, ← b.mk_eq_rank, Cardinal.lift_aleph0, Cardinal.lift_lt_aleph0,
    Cardinal.lt_aleph0_iff_fintype] at h


/-- If a module has a finite dimension, all bases are indexed by a finite type. -/
noncomputable def Basis.fintypeIndexOfRankLtAleph0 {ι : Type*} (b : Basis ι R M)
    (h : Module.rank R M < ℵ₀) : Fintype ι :=
  Classical.choice (b.nonempty_fintype_index_of_rank_lt_aleph0 h)


/-- If a module has a finite dimension, all bases are indexed by a finite set. -/
theorem Basis.finite_index_of_rank_lt_aleph0 {ι : Type*} {s : Set ι} (b : Basis s R M)
    (h : Module.rank R M < ℵ₀) : s.Finite :=
  finite_def.2 (b.nonempty_fintype_index_of_rank_lt_aleph0 h)


theorem cardinalMk_le_finrank [Module.Finite R M]
    {ι : Type w} {b : ι → M} (h : LinearIndependent R b) : #ι ≤ finrank R M := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    ι : Type w
    b : ι → M
    h : LinearIndependent R b
    ⊢ LE.le (Cardinal.mk ι) ↑(Module.finrank R M)
  -/
  rw [← lift_le.{max v w}]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    ι : Type w
    b : ι → M
    h : LinearIndependent R b
    ⊢ LE.le (Cardinal.lift.{max v w, w} (Cardinal.mk ι)) (Cardinal.lift.{max v w,  …
  -/
  simpa only [← finrank_eq_rank, lift_natCast, lift_le_nat_iff] using h.cardinal_lift_le_rank
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_le_finrank := cardinalMk_le_finrank


theorem fintype_card_le_finrank [Module.Finite R M]
    {ι : Type*} [Fintype ι] {b : ι → M} (h : LinearIndependent R b) :
    Fintype.card ι ≤ finrank R M := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    inst✝¹ : Module.Finite R M
    ι : Type u_1
    inst✝ : Fintype ι
    b : ι → M
    h : LinearIndependent R b
    ⊢ LE.le (Fintype.card ι) (Module.finrank R M)
  -/
  simpa using h.cardinalMk_le_finrank
  /-
    🎉 no goals
  -/


theorem finset_card_le_finrank [Module.Finite R M]
    {b : Finset M} (h : LinearIndependent R (fun x => x : b → M)) :
    b.card ≤ finrank R M := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    b : Finset M
    h : LinearIndependent R fun x => ↑x
    ⊢ LE.le b.card (Module.finrank R M)
  -/
  rw [← Fintype.card_coe]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    b : Finset M
    h : LinearIndependent R fun x => ↑x
    ⊢ LE.le (Fintype.card (Subtype fun x => Membership.mem b x)) (Module.finrank R …
  -/
  exact h.fintype_card_le_finrank
  /-
    🎉 no goals
  -/


theorem lt_aleph0_of_finite {ι : Type w}
    [Module.Finite R M] {v : ι → M} (h : LinearIndependent R v) : #ι < ℵ₀ := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    ι : Type w
    inst✝ : Module.Finite R M
    v : ι → M
    h : LinearIndependent R v
    ⊢ LT.lt (Cardinal.mk ι) Cardinal.aleph0
  -/
  apply Cardinal.lift_lt.1
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    ι : Type w
    inst✝ : Module.Finite R M
    v : ι → M
    h : LinearIndependent R v
    ⊢ LT.lt (Cardinal.lift.{?u.118366, w} (Cardinal.mk ι)) (Cardinal.lift.{?u.1183 …
  -/
  apply lt_of_le_of_lt
    /-
      case hab
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : StrongRankCondition R
      ι : Type w
      inst✝ : Module.Finite R M
      v : ι → M
      h : LinearIndependent R v
      ⊢ LE.le (Cardinal.lift.{?u.118366, w} (Cardinal.mk ι)) ?b
    -/
  · apply h.cardinal_lift_le_rank
    /-
      🎉 no goals
    -/
    /-
      case hbc
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : StrongRankCondition R
      ι : Type w
      inst✝ : Module.Finite R M
      v : ι → M
      h : LinearIndependent R v
      ⊢ LT.lt (Cardinal.lift.{w, v} (Module.rank R M)) (Cardinal.lift.{v, w} Cardina …
    -/
  · rw [← finrank_eq_rank, Cardinal.lift_aleph0, Cardinal.lift_natCast]
    /-
      case hbc
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : StrongRankCondition R
      ι : Type w
      inst✝ : Module.Finite R M
      v : ι → M
      h : LinearIndependent R v
      ⊢ LT.lt (↑(Module.finrank R M)) Cardinal.aleph0
    -/
    apply Cardinal.nat_lt_aleph0
    /-
      🎉 no goals
    -/


theorem finite [Module.Finite R M] {ι : Type*} {f : ι → M}
    (h : LinearIndependent R f) : Finite ι :=
  Cardinal.lt_aleph0_iff_finite.1 <| h.lt_aleph0_of_finite


theorem setFinite [Module.Finite R M] {b : Set M}
    (h : LinearIndependent R fun x : b => (x : M)) : b.Finite :=
  Cardinal.lt_aleph0_iff_set_finite.mp h.lt_aleph0_of_finite


lemma exists_set_linearIndependent_of_lt_rank {n : Cardinal} (hn : n < Module.rank R M) :
    ∃ s : Set M, #s = n ∧ LinearIndependent R ((↑) : s → M) := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Cardinal.{v}
    hn : LT.lt n (Module.rank R M)
    ⊢ Exists fun s => And (Eq (Cardinal.mk ↑s) n) (LinearIndependent R Subtype.val)
  -/
  obtain ⟨⟨s, hs⟩, hs'⟩ := exists_lt_of_lt_ciSup' (hn.trans_eq (Module.rank_def R M))
  /-
    case intro.mk
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Cardinal.{v}
    hn : LT.lt n (Module.rank R M)
    s : Set M
    hs : LinearIndependent R Subtype.val
    hs' : LT.lt n (Cardinal.mk ↑↑⟨s, hs⟩)
    ⊢ Exists fun s => And (Eq (Cardinal.mk ↑s) n) (LinearIndependent R Subtype.val)
  -/
  obtain ⟨t, ht, ht'⟩ := le_mk_iff_exists_subset.mp hs'.le
  /-
    case intro.mk.intro.intro
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Cardinal.{v}
    hn : LT.lt n (Module.rank R M)
    s : Set M
    hs : LinearIndependent R Subtype.val
    hs' : LT.lt n (Cardinal.mk ↑↑⟨s, hs⟩)
    t : Set M
    ht : HasSubset.Subset t ↑⟨s, hs⟩
    ht' : Eq (Cardinal.mk ↑t) n
    ⊢ Exists fun s => And (Eq (Cardinal.mk ↑s) n) (LinearIndependent R Subtype.val)
  -/
  exact ⟨t, ht', .mono ht hs⟩
  /-
    🎉 no goals
  -/


lemma exists_finset_linearIndependent_of_le_rank {n : ℕ} (hn : n ≤ Module.rank R M) :
    ∃ s : Finset M, s.card = n ∧ LinearIndependent R ((↑) : s → M) := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    hn : LE.le (↑n) (Module.rank R M)
    ⊢ Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
  -/
  have := nonempty_linearIndependent_set
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    hn : LE.le (↑n) (Module.rank R M)
    this : ∀ (R : Type ?u.131462) (M : Type ?u.131461) [inst : Semiring R] [inst_1 …
    ⊢ Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
  -/
  cases' hn.eq_or_lt with h h
  · obtain ⟨⟨s, hs⟩, hs'⟩ := Cardinal.exists_eq_natCast_of_iSup_eq _
      (Cardinal.bddAbove_range _) _ (h.trans (Module.rank_def R M)).symm
    /-
      case inl.intro.mk
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      hn : LE.le (↑n) (Module.rank R M)
      this : ∀ (R : Type u) (M : Type v) [inst : Semiring R] [inst_1 : AddCommMonoid …
      h : Eq (↑n) (Module.rank R M)
      s : Set M
      hs : LinearIndependent R Subtype.val
      hs' : Eq (Cardinal.mk ↑↑⟨s, hs⟩) ↑n
      ⊢ Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
    -/
    have : Finite s := lt_aleph0_iff_finite.mp (hs' ▸ nat_lt_aleph0 n)
    /-
      case inl.intro.mk
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      hn : LE.le (↑n) (Module.rank R M)
      this✝ : ∀ (R : Type u) (M : Type v) [inst : Semiring R] [inst_1 : AddCommMonoi …
      h : Eq (↑n) (Module.rank R M)
      s : Set M
      hs : LinearIndependent R Subtype.val
      hs' : Eq (Cardinal.mk ↑↑⟨s, hs⟩) ↑n
      this : Finite ↑s
      ⊢ Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
    -/
    cases nonempty_fintype s
    /-
      case inl.intro.mk.intro
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      hn : LE.le (↑n) (Module.rank R M)
      this✝ : ∀ (R : Type u) (M : Type v) [inst : Semiring R] [inst_1 : AddCommMonoi …
      h : Eq (↑n) (Module.rank R M)
      s : Set M
      hs : LinearIndependent R Subtype.val
      hs' : Eq (Cardinal.mk ↑↑⟨s, hs⟩) ↑n
      this : Finite ↑s
      val✝ : Fintype ↑s
      ⊢ Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
    -/
    exact ⟨s.toFinset, by simpa using hs', by convert hs using 3 <;> exact Set.mem_toFinset⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      hn : LE.le (↑n) (Module.rank R M)
      this : ∀ (R : Type u) (M : Type v) [inst : Semiring R] [inst_1 : AddCommMonoid …
      h : LT.lt (↑n) (Module.rank R M)
      ⊢ Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
    -/
  · obtain ⟨s, hs, hs'⟩ := exists_set_linearIndependent_of_lt_rank h
    /-
      case inr.intro.intro
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      hn : LE.le (↑n) (Module.rank R M)
      this : ∀ (R : Type u) (M : Type v) [inst : Semiring R] [inst_1 : AddCommMonoid …
      h : LT.lt (↑n) (Module.rank R M)
      s : Set M
      hs : Eq (Cardinal.mk ↑s) ↑n
      hs' : LinearIndependent R Subtype.val
      ⊢ Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
    -/
    have : Finite s := lt_aleph0_iff_finite.mp (hs ▸ nat_lt_aleph0 n)
    /-
      case inr.intro.intro
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      hn : LE.le (↑n) (Module.rank R M)
      this✝ : ∀ (R : Type u) (M : Type v) [inst : Semiring R] [inst_1 : AddCommMonoi …
      h : LT.lt (↑n) (Module.rank R M)
      s : Set M
      hs : Eq (Cardinal.mk ↑s) ↑n
      hs' : LinearIndependent R Subtype.val
      this : Finite ↑s
      ⊢ Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
    -/
    cases nonempty_fintype s
    /-
      case inr.intro.intro.intro
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      hn : LE.le (↑n) (Module.rank R M)
      this✝ : ∀ (R : Type u) (M : Type v) [inst : Semiring R] [inst_1 : AddCommMonoi …
      h : LT.lt (↑n) (Module.rank R M)
      s : Set M
      hs : Eq (Cardinal.mk ↑s) ↑n
      hs' : LinearIndependent R Subtype.val
      this : Finite ↑s
      val✝ : Fintype ↑s
      ⊢ Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
    -/
    exact ⟨s.toFinset, by simpa using hs, by convert hs' using 3 <;> exact Set.mem_toFinset⟩
    /-
      🎉 no goals
    -/


lemma exists_linearIndependent_of_le_rank {n : ℕ} (hn : n ≤ Module.rank R M) :
    ∃ f : Fin n → M, LinearIndependent R f :=
  have ⟨_, hs, hs'⟩ := exists_finset_linearIndependent_of_le_rank hn
  ⟨_, (linearIndependent_equiv (Finset.equivFinOfCardEq hs).symm).mpr hs'⟩


lemma natCast_le_rank_iff [Nontrivial R] {n : ℕ} :
    n ≤ Module.rank R M ↔ ∃ f : Fin n → M, LinearIndependent R f :=
  ⟨exists_linearIndependent_of_le_rank,
               /-
                 R : Type u
                 M : Type v
                 inst✝³ : Ring R
                 inst✝² : AddCommGroup M
                 inst✝¹ : Module R M
                 inst✝ : Nontrivial R
                 n : Nat
                 H : Exists fun f => LinearIndependent R f
                 ⊢ LE.le (↑n) (Module.rank R M)
               -/
    fun H ↦ by simpa using H.choose_spec.cardinal_lift_le_rank⟩
               /-
                 🎉 no goals
               -/


lemma natCast_le_rank_iff_finset [Nontrivial R] {n : ℕ} :
    n ≤ Module.rank R M ↔ ∃ s : Finset M, s.card = n ∧ LinearIndependent R ((↑) : s → M) :=
  ⟨exists_finset_linearIndependent_of_le_rank,
                         /-
                           R : Type u
                           M : Type v
                           inst✝³ : Ring R
                           inst✝² : AddCommGroup M
                           inst✝¹ : Module R M
                           inst✝ : Nontrivial R
                           n : Nat
                           x✝ : Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
                           s : Finset M
                           h₁ : Eq s.card n
                           h₂ : LinearIndependent R Subtype.val
                           ⊢ LE.le (↑n) (Module.rank R M)
                         -/
    fun ⟨s, h₁, h₂⟩ ↦ by simpa [h₁] using h₂.cardinal_le_rank⟩
                         /-
                           🎉 no goals
                         -/


lemma exists_finset_linearIndependent_of_le_finrank {n : ℕ} (hn : n ≤ finrank R M) :
    ∃ s : Finset M, s.card = n ∧ LinearIndependent R ((↑) : s → M) := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    hn : LE.le n (Module.finrank R M)
    ⊢ Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
  -/
  by_cases h : finrank R M = 0
    /-
      case pos
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      hn : LE.le n (Module.finrank R M)
      h : Eq (Module.finrank R M) 0
      ⊢ Exists fun s => And (Eq s.card n) (LinearIndependent R Subtype.val)
    -/
  · rw [le_zero_iff.mp (hn.trans_eq h)]
    /-
      case pos
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Nat
      hn : LE.le n (Module.finrank R M)
      h : Eq (Module.finrank R M) 0
      ⊢ Exists fun s => And (Eq s.card 0) (LinearIndependent R Subtype.val)
    -/
    exact ⟨∅, rfl, by convert linearIndependent_empty R M using 2 <;> aesop⟩
    /-
      🎉 no goals
    -/
  exact exists_finset_linearIndependent_of_le_rank
    ((Nat.cast_le.mpr hn).trans_eq (cast_toNat_of_lt_aleph0 (toNat_ne_zero.mp h).2))


lemma exists_linearIndependent_of_le_finrank {n : ℕ} (hn : n ≤ finrank R M) :
    ∃ f : Fin n → M, LinearIndependent R f :=
  have ⟨_, hs, hs'⟩ := exists_finset_linearIndependent_of_le_finrank hn
  ⟨_, (linearIndependent_equiv (Finset.equivFinOfCardEq hs).symm).mpr hs'⟩


variable [Module.Finite R M] [StrongRankCondition R] in
theorem Module.Finite.not_linearIndependent_of_infinite {ι : Type*} [Infinite ι]
    (v : ι → M) : ¬LinearIndependent R v := mt LinearIndependent.finite <| @not_finite _ _


theorem iSupIndep.subtype_ne_bot_le_rank [Nontrivial R]
    {V : ι → Submodule R M} (hV : iSupIndep V) :
    Cardinal.lift.{v} #{ i : ι // V i ≠ ⊥ } ≤ Cardinal.lift.{w} (Module.rank R M) := by
  /-
    R : Type u
    M : Type v
    ι : Type w
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Nontrivial R
    V : ι → Submodule R M
    hV : iSupIndep V
    ⊢ LE.le (Cardinal.lift.{v, w} (Cardinal.mk (Subtype fun i => Ne (V i) Bot.bot) …
  -/
  set I := { i : ι // V i ≠ ⊥ }
  have hI : ∀ i : I, ∃ v ∈ V i, v ≠ (0 : M) := by
    intro i
    rw [← Submodule.ne_bot_iff]
    exact i.prop
  /-
    R : Type u
    M : Type v
    ι : Type w
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Nontrivial R
    V : ι → Submodule R M
    hV : iSupIndep V
    I : Type w := Subtype fun i => Ne (V i) Bot.bot
    hI : ∀ (i : I), Exists fun v => And (Membership.mem (V ↑i) v) (Ne v 0)
    ⊢ LE.le (Cardinal.lift.{v, w} (Cardinal.mk I)) (Cardinal.lift.{w, v} (Module.r …
  -/
  choose v hvV hv using hI
  /-
    R : Type u
    M : Type v
    ι : Type w
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Nontrivial R
    V : ι → Submodule R M
    hV : iSupIndep V
    I : Type w := Subtype fun i => Ne (V i) Bot.bot
    v : I → M
    hvV : ∀ (i : I), Membership.mem (V ↑i) (v i)
    hv : ∀ (i : I), Ne (v i) 0
    ⊢ LE.le (Cardinal.lift.{v, w} (Cardinal.mk I)) (Cardinal.lift.{w, v} (Module.r …
  -/
  have : LinearIndependent R v := (hV.comp Subtype.coe_injective).linearIndependent _ hvV hv
  /-
    R : Type u
    M : Type v
    ι : Type w
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Nontrivial R
    V : ι → Submodule R M
    hV : iSupIndep V
    I : Type w := Subtype fun i => Ne (V i) Bot.bot
    v : I → M
    hvV : ∀ (i : I), Membership.mem (V ↑i) (v i)
    hv : ∀ (i : I), Ne (v i) 0
    this : LinearIndependent R v
    ⊢ LE.le (Cardinal.lift.{v, w} (Cardinal.mk I)) (Cardinal.lift.{w, v} (Module.r …
  -/
  exact this.cardinal_lift_le_rank
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.Independent.subtype_ne_bot_le_rank := iSupIndep.subtype_ne_bot_le_rank


theorem iSupIndep.subtype_ne_bot_le_finrank_aux
    {p : ι → Submodule R M} (hp : iSupIndep p) :
    #{ i // p i ≠ ⊥ } ≤ (finrank R M : Cardinal.{w}) := by
  suffices Cardinal.lift.{v} #{ i // p i ≠ ⊥ } ≤ Cardinal.lift.{v} (finrank R M : Cardinal.{w}) by
    rwa [Cardinal.lift_le] at this
  calc
    Cardinal.lift.{v} #{ i // p i ≠ ⊥ } ≤ Cardinal.lift.{w} (Module.rank R M) :=
      hp.subtype_ne_bot_le_rank
    _ = Cardinal.lift.{w} (finrank R M : Cardinal.{v}) := by rw [finrank_eq_rank]
    _ = Cardinal.lift.{v} (finrank R M : Cardinal.{w}) := by simp


/-- If `p` is an independent family of submodules of a `R`-finite module `M`, then the
number of nontrivial subspaces in the family `p` is finite. -/
noncomputable def iSupIndep.fintypeNeBotOfFiniteDimensional
    {p : ι → Submodule R M} (hp : iSupIndep p) :
    Fintype { i : ι // p i ≠ ⊥ } := by
  suffices #{ i // p i ≠ ⊥ } < (ℵ₀ : Cardinal.{w}) by
    rw [Cardinal.lt_aleph0_iff_fintype] at this
    exact this.some
  /-
    R : Type u
    M M₁ : Type v
    M' : Type v'
    ι : Type w
    inst✝⁹ : Ring R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup M'
    inst✝⁶ : AddCommGroup M₁
    inst✝⁵ : Module R M
    inst✝⁴ : Module R M'
    inst✝³ : Module R M₁
    inst✝² : NoZeroSMulDivisors R M
    inst✝¹ : Module.Finite R M
    inst✝ : StrongRankCondition R
    p : ι → Submodule R M
    hp : iSupIndep p
    ⊢ LT.lt (Cardinal.mk (Subtype fun i => Ne (p i) Bot.bot)) Cardinal.aleph0
  -/
  refine lt_of_le_of_lt hp.subtype_ne_bot_le_finrank_aux ?_
  /-
    R : Type u
    M M₁ : Type v
    M' : Type v'
    ι : Type w
    inst✝⁹ : Ring R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup M'
    inst✝⁶ : AddCommGroup M₁
    inst✝⁵ : Module R M
    inst✝⁴ : Module R M'
    inst✝³ : Module R M₁
    inst✝² : NoZeroSMulDivisors R M
    inst✝¹ : Module.Finite R M
    inst✝ : StrongRankCondition R
    p : ι → Submodule R M
    hp : iSupIndep p
    ⊢ LT.lt (↑(Module.finrank R M)) Cardinal.aleph0
  -/
  simp [Cardinal.nat_lt_aleph0]
  /-
    🎉 no goals
  -/


/-- If `p` is an independent family of submodules of a `R`-finite module `M`, then the
number of nontrivial subspaces in the family `p` is bounded above by the dimension of `M`.

Note that the `Fintype` hypothesis required here can be provided by
`iSupIndep.fintypeNeBotOfFiniteDimensional`. -/
theorem iSupIndep.subtype_ne_bot_le_finrank
    {p : ι → Submodule R M} (hp : iSupIndep p) [Fintype { i // p i ≠ ⊥ }] :
                                                      /-
                                                        R : Type u
                                                        M : Type v
                                                        ι : Type w
                                                        inst✝⁶ : Ring R
                                                        inst✝⁵ : AddCommGroup M
                                                        inst✝⁴ : Module R M
                                                        inst✝³ : NoZeroSMulDivisors R M
                                                        inst✝² : Module.Finite R M
                                                        inst✝¹ : StrongRankCondition R
                                                        p : ι → Submodule R M
                                                        hp : iSupIndep p
                                                        inst✝ : Fintype (Subtype fun i => Ne (p i) Bot.bot)
                                                        ⊢ LE.le (Fintype.card (Subtype fun i => Ne (p i) Bot.bot)) (Module.finrank R M)
                                                      -/
    Fintype.card { i // p i ≠ ⊥ } ≤ finrank R M := by simpa using hp.subtype_ne_bot_le_finrank_aux
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- If a finset has cardinality larger than the rank of a module,
then there is a nontrivial linear relation amongst its elements. -/
theorem Module.exists_nontrivial_relation_of_finrank_lt_card {t : Finset M}
    (h : finrank R M < t.card) : ∃ f : M → R, ∑ e ∈ t, f e • e = 0 ∧ ∃ x ∈ t, f x ≠ 0 := by
  obtain ⟨g, sum, z, nonzero⟩ := Fintype.not_linearIndependent_iff.mp
    (mt LinearIndependent.finset_card_le_finrank h.not_le)
  /-
    case intro.intro.intro
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : StrongRankCondition R
    t : Finset M
    h : LT.lt (Module.finrank R M) t.card
    g : (Subtype fun x => Membership.mem t x) → R
    sum : Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) ↑i) 0
    z : Subtype fun x => Membership.mem t x
    nonzero : Ne (g z) 0
    ⊢ Exists fun f => And (Eq (t.sum fun e => HSMul.hSMul (f e) e) 0) (Exists fun  …
  -/
  refine ⟨Subtype.val.extend g 0, ?_, z, z.2, by rwa [Subtype.val_injective.extend_apply]⟩
  /-
    case intro.intro.intro
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : StrongRankCondition R
    t : Finset M
    h : LT.lt (Module.finrank R M) t.card
    g : (Subtype fun x => Membership.mem t x) → R
    sum : Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) ↑i) 0
    z : Subtype fun x => Membership.mem t x
    nonzero : Ne (g z) 0
    ⊢ Eq (t.sum fun e => HSMul.hSMul (Function.extend Subtype.val g 0 e) e) 0
  -/
  rw [← Finset.sum_finset_coe]; convert sum; apply Subtype.val_injective.extend_apply
                                             /-
                                               🎉 no goals
                                             -/


/-- If a finset has cardinality larger than `finrank + 1`,
then there is a nontrivial linear relation amongst its elements,
such that the coefficients of the relation sum to zero. -/
theorem Module.exists_nontrivial_relation_sum_zero_of_finrank_succ_lt_card
    {t : Finset M} (h : finrank R M + 1 < t.card) :
    ∃ f : M → R, ∑ e ∈ t, f e • e = 0 ∧ ∑ e ∈ t, f e = 0 ∧ ∃ x ∈ t, f x ≠ 0 := by
  -- Pick an element x₀ ∈ t,
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : StrongRankCondition R
    t : Finset M
    h : LT.lt (HAdd.hAdd (Module.finrank R M) 1) t.card
    ⊢ Exists fun f => And (Eq (t.sum fun e => HSMul.hSMul (f e) e) 0) (And (Eq (t. …
  -/
  obtain ⟨x₀, x₀_mem⟩ := card_pos.1 ((Nat.succ_pos _).trans h)
  -- and apply the previous lemma to the {xᵢ - x₀}
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : StrongRankCondition R
    t : Finset M
    h : LT.lt (HAdd.hAdd (Module.finrank R M) 1) t.card
    x₀ : M
    x₀_mem : Membership.mem t x₀
    ⊢ Exists fun f => And (Eq (t.sum fun e => HSMul.hSMul (f e) e) 0) (And (Eq (t. …
  -/
  let shift : M ↪ M := ⟨(· - x₀), sub_left_injective⟩
  classical
  let t' := (t.erase x₀).map shift
  have h' : finrank R M < t'.card := by
    rw [card_map, card_erase_of_mem x₀_mem]
    exact Nat.lt_pred_iff.mpr h
  -- to obtain a function `g`.
  obtain ⟨g, gsum, x₁, x₁_mem, nz⟩ := exists_nontrivial_relation_of_finrank_lt_card h'
  -- Then obtain `f` by translating back by `x₀`,
  -- and setting the value of `f` at `x₀` to ensure `∑ e ∈ t, f e = 0`.
  let f : M → R := fun z ↦ if z = x₀ then -∑ z ∈ t.erase x₀, g (z - x₀) else g (z - x₀)
  refine ⟨f, ?_, ?_, ?_⟩
  -- After this, it's a matter of verifying the properties,
  -- based on the corresponding properties for `g`.
  · rw [sum_map, Embedding.coeFn_mk] at gsum
    simp_rw [f, ← t.sum_erase_add _ x₀_mem, if_pos, neg_smul, sum_smul,
             ← sub_eq_add_neg, ← sum_sub_distrib, ← gsum, smul_sub]
    refine sum_congr rfl fun x x_mem ↦ ?_
    rw [if_neg (mem_erase.mp x_mem).1]
  · simp_rw [f, ← t.sum_erase_add _ x₀_mem, if_pos, add_neg_eq_zero]
    exact sum_congr rfl fun x x_mem ↦ if_neg (mem_erase.mp x_mem).1
  · obtain ⟨x₁, x₁_mem', rfl⟩ := Finset.mem_map.mp x₁_mem
    have := mem_erase.mp x₁_mem'
    exact ⟨x₁, by
      simpa only [f, Embedding.coeFn_mk, sub_add_cancel, this.2, true_and, if_neg this.1]⟩


/-- A (finite dimensional) space that is a subsingleton has zero `finrank`. -/
@[nontriviality]
theorem Module.finrank_zero_of_subsingleton [Subsingleton M] :
    finrank R M = 0 := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Nontrivial R
    inst✝ : Subsingleton M
    ⊢ Eq (Module.finrank R M) 0
  -/
  rw [finrank, rank_subsingleton', _root_.map_zero]
  /-
    🎉 no goals
  -/


lemma LinearIndependent.finrank_eq_zero_of_infinite {ι} [Infinite ι] {v : ι → M}
    (hv : LinearIndependent R v) : finrank R M = 0 := toNat_eq_zero.mpr <| .inr hv.aleph0_le_rank


/-- A finite dimensional space is nontrivial if it has positive `finrank`. -/
theorem Module.nontrivial_of_finrank_pos (h : 0 < finrank R M) : Nontrivial M :=
  rank_pos_iff_nontrivial.mp (lt_rank_of_lt_finrank h)


/-- A finite dimensional space is nontrivial if it has `finrank` equal to the successor of a
natural number. -/
theorem Module.nontrivial_of_finrank_eq_succ {n : ℕ}
    (hn : finrank R M = n.succ) : Nontrivial M :=
                                         /-
                                           R : Type u
                                           M : Type v
                                           inst✝⁴ : Ring R
                                           inst✝³ : AddCommGroup M
                                           inst✝² : Module R M
                                           inst✝¹ : Nontrivial R
                                           inst✝ : NoZeroSMulDivisors R M
                                           n : Nat
                                           hn : Eq (Module.finrank R M) n.succ
                                           ⊢ LT.lt 0 (Module.finrank R M)
                                         -/
  nontrivial_of_finrank_pos (R := R) (by rw [hn]; exact n.succ_pos)
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem finrank_bot : finrank R (⊥ : Submodule R M) = 0 :=
  finrank_eq_of_rank_eq (rank_bot _ _)


/-- A finite rank torsion-free module has positive `finrank` iff it has a nonzero element. -/
theorem Module.finrank_pos_iff_exists_ne_zero [NoZeroSMulDivisors R M] :
    0 < finrank R M ↔ ∃ x : M, x ≠ 0 := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    inst✝¹ : Module.Finite R M
    inst✝ : NoZeroSMulDivisors R M
    ⊢ Iff (LT.lt 0 (Module.finrank R M)) (Exists fun x => Ne x 0)
  -/
  rw [← @rank_pos_iff_exists_ne_zero R M, ← finrank_eq_rank]
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    inst✝¹ : Module.Finite R M
    inst✝ : NoZeroSMulDivisors R M
    ⊢ Iff (LT.lt 0 (Module.finrank R M)) (LT.lt 0 ↑(Module.finrank R M))
  -/
  norm_cast
  /-
    🎉 no goals
  -/


/-- An `R`-finite torsion-free module has positive `finrank` iff it is nontrivial. -/
theorem Module.finrank_pos_iff [NoZeroSMulDivisors R M] :
    0 < finrank R M ↔ Nontrivial M := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    inst✝¹ : Module.Finite R M
    inst✝ : NoZeroSMulDivisors R M
    ⊢ Iff (LT.lt 0 (Module.finrank R M)) (Nontrivial M)
  -/
  rw [← rank_pos_iff_nontrivial (R := R), ← finrank_eq_rank]
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    inst✝¹ : Module.Finite R M
    inst✝ : NoZeroSMulDivisors R M
    ⊢ Iff (LT.lt 0 (Module.finrank R M)) (LT.lt 0 ↑(Module.finrank R M))
  -/
  norm_cast
  /-
    🎉 no goals
  -/


/-- A nontrivial finite dimensional space has positive `finrank`. -/
theorem Module.finrank_pos [NoZeroSMulDivisors R M] [h : Nontrivial M] :
    0 < finrank R M :=
  finrank_pos_iff.mpr h


/-- See `Module.finrank_zero_iff`
  for the stronger version with `NoZeroSMulDivisors R M`. -/
theorem Module.finrank_eq_zero_iff :
    finrank R M = 0 ↔ ∀ x : M, ∃ a : R, a ≠ 0 ∧ a • x = 0 := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    ⊢ Iff (Eq (Module.finrank R M) 0) (∀ (x : M), Exists fun a => And (Ne a 0) (Eq …
  -/
  rw [← rank_eq_zero_iff (R := R), ← finrank_eq_rank]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    ⊢ Iff (Eq (Module.finrank R M) 0) (Eq (↑(Module.finrank R M)) 0)
  -/
  norm_cast
  /-
    🎉 no goals
  -/


/-- A finite dimensional space has zero `finrank` iff it is a subsingleton.
This is the `finrank` version of `rank_zero_iff`. -/
theorem Module.finrank_zero_iff [NoZeroSMulDivisors R M] :
    finrank R M = 0 ↔ Subsingleton M := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    inst✝¹ : Module.Finite R M
    inst✝ : NoZeroSMulDivisors R M
    ⊢ Iff (Eq (Module.finrank R M) 0) (Subsingleton M)
  -/
  rw [← rank_zero_iff (R := R), ← finrank_eq_rank]
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    inst✝¹ : Module.Finite R M
    inst✝ : NoZeroSMulDivisors R M
    ⊢ Iff (Eq (Module.finrank R M) 0) (Eq (↑(Module.finrank R M)) 0)
  -/
  norm_cast
  /-
    🎉 no goals
  -/


/-- Similar to `rank_quotient_add_rank_le` but for `finrank` and a finite `M`. -/
lemma Module.finrank_quotient_add_finrank_le (N : Submodule R M) :
    finrank R (M ⧸ N) + finrank R N ≤ finrank R M := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    N : Submodule R M
    ⊢ LE.le (HAdd.hAdd (Module.finrank R (HasQuotient.Quotient M N)) (Module.finra …
  -/
  haveI := nontrivial_of_invariantBasisNumber R
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    N : Submodule R M
    this : Nontrivial R
    ⊢ LE.le (HAdd.hAdd (Module.finrank R (HasQuotient.Quotient M N)) (Module.finra …
  -/
  have := rank_quotient_add_rank_le N
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    N : Submodule R M
    this✝ : Nontrivial R
    this : LE.le (HAdd.hAdd (Module.rank R (HasQuotient.Quotient M N)) (Module.ran …
    ⊢ LE.le (HAdd.hAdd (Module.finrank R (HasQuotient.Quotient M N)) (Module.finra …
  -/
  rw [← finrank_eq_rank R M, ← finrank_eq_rank R, ← N.finrank_eq_rank] at this
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    N : Submodule R M
    this✝ : Nontrivial R
    this : LE.le (HAdd.hAdd ↑(Module.finrank R (HasQuotient.Quotient M N)) ↑(Modul …
    ⊢ LE.le (HAdd.hAdd (Module.finrank R (HasQuotient.Quotient M N)) (Module.finra …
  -/
  exact mod_cast this
  /-
    🎉 no goals
  -/


theorem Module.finrank_eq_zero_of_rank_eq_zero (h : Module.rank R M = 0) :
    finrank R M = 0 := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : Eq (Module.rank R M) 0
    ⊢ Eq (Module.finrank R M) 0
  -/
  delta finrank
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : Eq (Module.rank R M) 0
    ⊢ Eq (Cardinal.toNat (Module.rank R M)) 0
  -/
  rw [h, zero_toNat]
  /-
    🎉 no goals
  -/


theorem Submodule.bot_eq_top_of_rank_eq_zero [NoZeroSMulDivisors R M] (h : Module.rank R M = 0) :
    (⊥ : Submodule R M) = ⊤ := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    h : Eq (Module.rank R M) 0
    ⊢ Eq Bot.bot Top.top
  -/
  nontriviality R
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    h : Eq (Module.rank R M) 0
    a✝ : Nontrivial R
    ⊢ Eq Bot.bot Top.top
  -/
  rw [rank_zero_iff] at h
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    h : Subsingleton M
    a✝ : Nontrivial R
    ⊢ Eq Bot.bot Top.top
  -/
  subsingleton
  /-
    🎉 no goals
  -/


/-- See `rank_subsingleton` for the reason that `Nontrivial R` is needed. -/
@[simp]
theorem Submodule.rank_eq_zero [Nontrivial R] [NoZeroSMulDivisors R M] {S : Submodule R M} :
    Module.rank R S = 0 ↔ S = ⊥ :=
  ⟨fun h =>
    (Submodule.eq_bot_iff _).2 fun x hx =>
      congr_arg Subtype.val <|
        ((Submodule.eq_bot_iff _).1 <| Eq.symm <| Submodule.bot_eq_top_of_rank_eq_zero h) ⟨x, hx⟩
          Submodule.mem_top,
                /-
                  R : Type u
                  M : Type v
                  inst✝⁴ : Ring R
                  inst✝³ : AddCommGroup M
                  inst✝² : Module R M
                  inst✝¹ : Nontrivial R
                  inst✝ : NoZeroSMulDivisors R M
                  S : Submodule R M
                  h : Eq S Bot.bot
                  ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem S x)) 0
                -/
    fun h => by rw [h, rank_bot]⟩
                /-
                  🎉 no goals
                -/


@[simp]
theorem Submodule.finrank_eq_zero [StrongRankCondition R] [NoZeroSMulDivisors R M]
    {S : Submodule R M} [Module.Finite R S] :
    finrank R S = 0 ↔ S = ⊥ := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    inst✝¹ : NoZeroSMulDivisors R M
    S : Submodule R M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem S x)
    ⊢ Iff (Eq (Module.finrank R (Subtype fun x => Membership.mem S x)) 0) (Eq S Bo …
  -/
  rw [← Submodule.rank_eq_zero, ← finrank_eq_rank, ← @Nat.cast_zero Cardinal, Nat.cast_inj]
  /-
    🎉 no goals
  -/


@[simp]
lemma Submodule.one_le_finrank_iff [StrongRankCondition R] [NoZeroSMulDivisors R M]
    {S : Submodule R M} [Module.Finite R S] :
    1 ≤ finrank R S ↔ S ≠ ⊥ := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    inst✝¹ : NoZeroSMulDivisors R M
    S : Submodule R M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem S x)
    ⊢ Iff (LE.le 1 (Module.finrank R (Subtype fun x => Membership.mem S x))) (Ne S …
  -/
  simp [← not_iff_not]
  /-
    🎉 no goals
  -/


theorem finrank_eq_zero_of_basis_imp_not_finite
    (h : ∀ s : Set M, Basis.{v} (s : Set M) R M → ¬s.Finite) : finrank R M = 0 := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    h : ∀ (s : Set M), Basis (↑s) R M → Not s.Finite
    ⊢ Eq (Module.finrank R M) 0
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module.Free R M
      h : ∀ (s : Set M), Basis (↑s) R M → Not s.Finite
      h✝ : Subsingleton R
      ⊢ Eq (Module.finrank R M) 0
    -/
  · have := Module.subsingleton R M
    /-
      case inl
      R : Type u
      M : Type v
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module.Free R M
      h : ∀ (s : Set M), Basis (↑s) R M → Not s.Finite
      h✝ : Subsingleton R
      this : Subsingleton M
      ⊢ Eq (Module.finrank R M) 0
    -/
    exact (h ∅ ⟨LinearEquiv.ofSubsingleton _ _⟩ Set.finite_empty).elim
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    h : ∀ (s : Set M), Basis (↑s) R M → Not s.Finite
    h✝ : Nontrivial R
    ⊢ Eq (Module.finrank R M) 0
  -/
  obtain ⟨_, ⟨b⟩⟩ := (Module.free_iff_set R M).mp ‹_›
  /-
    case inr.intro.intro
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    h : ∀ (s : Set M), Basis (↑s) R M → Not s.Finite
    h✝ : Nontrivial R
    w✝ : Set M
    b : Basis (↑w✝) R M
    ⊢ Eq (Module.finrank R M) 0
  -/
  have := Set.Infinite.to_subtype (h _ b)
  /-
    case inr.intro.intro
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    h : ∀ (s : Set M), Basis (↑s) R M → Not s.Finite
    h✝ : Nontrivial R
    w✝ : Set M
    b : Basis (↑w✝) R M
    this : Infinite ↑w✝
    ⊢ Eq (Module.finrank R M) 0
  -/
  exact b.linearIndependent.finrank_eq_zero_of_infinite
  /-
    🎉 no goals
  -/


theorem finrank_eq_zero_of_basis_imp_false (h : ∀ s : Finset M, Basis.{v} (s : Set M) R M → False) :
    finrank R M = 0 :=
  finrank_eq_zero_of_basis_imp_not_finite fun s b hs =>
    h hs.toFinset
      (by
        /-
          R : Type u
          M : Type v
          inst✝³ : Ring R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.Free R M
          h : ∀ (s : Finset M), Basis (↑↑s) R M → False
          s : Set M
          b : Basis (↑s) R M
          hs : s.Finite
          ⊢ Basis (↑↑hs.toFinset) R M
        -/
        convert b
        /-
          case h.e'_1.h.e'_2
          R : Type u
          M : Type v
          inst✝³ : Ring R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.Free R M
          h : ∀ (s : Finset M), Basis (↑↑s) R M → False
          s : Set M
          b : Basis (↑s) R M
          hs : s.Finite
          ⊢ Eq (↑hs.toFinset) s
        -/
        simp)
        /-
          🎉 no goals
        -/


theorem finrank_eq_zero_of_not_exists_basis
    (h : ¬∃ s : Finset M, Nonempty (Basis (s : Set M) R M)) : finrank R M = 0 :=
  finrank_eq_zero_of_basis_imp_false fun s b => h ⟨s, ⟨b⟩⟩


theorem finrank_eq_zero_of_not_exists_basis_finite
    (h : ¬∃ (s : Set M) (_ : Basis.{v} (s : Set M) R M), s.Finite) : finrank R M = 0 :=
  finrank_eq_zero_of_basis_imp_not_finite fun s b hs => h ⟨s, b, hs⟩


theorem finrank_eq_zero_of_not_exists_basis_finset (h : ¬∃ s : Finset M, Nonempty (Basis s R M)) :
    finrank R M = 0 :=
  finrank_eq_zero_of_basis_imp_false fun s b => h ⟨s, ⟨b⟩⟩


/-- If there is a nonzero vector and every other vector is a multiple of it,
then the module has dimension one. -/
theorem rank_eq_one (v : M) (n : v ≠ 0) (h : ∀ w : M, ∃ c : R, c • v = w) :
    Module.rank R M = 1 := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : StrongRankCondition R
    v : M
    n : Ne v 0
    h : ∀ (w : M), Exists fun c => Eq (HSMul.hSMul c v) w
    ⊢ Eq (Module.rank R M) 1
  -/
  haveI := nontrivial_of_invariantBasisNumber R
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : StrongRankCondition R
    v : M
    n : Ne v 0
    h : ∀ (w : M), Exists fun c => Eq (HSMul.hSMul c v) w
    this : Nontrivial R
    ⊢ Eq (Module.rank R M) 1
  -/
  obtain ⟨b⟩ := (Basis.basis_singleton_iff.{_, _, u} PUnit).mpr ⟨v, n, h⟩
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : StrongRankCondition R
    v : M
    n : Ne v 0
    h : ∀ (w : M), Exists fun c => Eq (HSMul.hSMul c v) w
    this : Nontrivial R
    b : Basis PUnit.{u + 1} R M
    ⊢ Eq (Module.rank R M) 1
  -/
  rw [rank_eq_card_basis b, Fintype.card_punit, Nat.cast_one]
  /-
    🎉 no goals
  -/


/-- If there is a nonzero vector and every other vector is a multiple of it,
then the module has dimension one. -/
theorem finrank_eq_one (v : M) (n : v ≠ 0) (h : ∀ w : M, ∃ c : R, c • v = w) : finrank R M = 1 :=
  finrank_eq_of_rank_eq (rank_eq_one v n h)


/-- If every vector is a multiple of some `v : M`, then `M` has dimension at most one.
-/
theorem finrank_le_one (v : M) (h : ∀ w : M, ∃ c : R, c • v = w) : finrank R M ≤ 1 := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : StrongRankCondition R
    v : M
    h : ∀ (w : M), Exists fun c => Eq (HSMul.hSMul c v) w
    ⊢ LE.le (Module.finrank R M) 1
  -/
  haveI := nontrivial_of_invariantBasisNumber R
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : StrongRankCondition R
    v : M
    h : ∀ (w : M), Exists fun c => Eq (HSMul.hSMul c v) w
    this : Nontrivial R
    ⊢ LE.le (Module.finrank R M) 1
  -/
  rcases eq_or_ne v 0 with (rfl | hn)
  · haveI :=
      _root_.subsingleton_of_forall_eq (0 : M) fun w => by
        obtain ⟨c, rfl⟩ := h w
        simp
    /-
      case inl
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : StrongRankCondition R
      this✝ : Nontrivial R
      h : ∀ (w : M), Exists fun c => Eq (HSMul.hSMul c 0) w
      this : Subsingleton M
      ⊢ LE.le (Module.finrank R M) 1
    -/
    rw [finrank_zero_of_subsingleton]
    /-
      case inl
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : StrongRankCondition R
      this✝ : Nontrivial R
      h : ∀ (w : M), Exists fun c => Eq (HSMul.hSMul c 0) w
      this : Subsingleton M
      ⊢ LE.le 0 1
    -/
    exact zero_le_one
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : StrongRankCondition R
      v : M
      h : ∀ (w : M), Exists fun c => Eq (HSMul.hSMul c v) w
      this : Nontrivial R
      hn : Ne v 0
      ⊢ LE.le (Module.finrank R M) 1
    -/
  · exact (finrank_eq_one v hn h).le
    /-
      🎉 no goals
    -/


