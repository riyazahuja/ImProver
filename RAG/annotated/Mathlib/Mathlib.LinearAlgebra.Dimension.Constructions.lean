theorem LinearIndependent.sum_elim_of_quotient
    {M' : Submodule R M} {ι₁ ι₂} {f : ι₁ → M'} (hf : LinearIndependent R f) (g : ι₂ → M)
    (hg : LinearIndependent R (Submodule.Quotient.mk (p := M') ∘ g)) :
      LinearIndependent R (Sum.elim (f · : ι₁ → M) g) := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    M' : Submodule R M
    ι₁ : Type u_2
    ι₂ : Type u_3
    f : ι₁ → Subtype fun x => Membership.mem M' x
    hf : LinearIndependent R f
    g : ι₂ → M
    hg : LinearIndependent R (Function.comp Submodule.Quotient.mk g)
    ⊢ LinearIndependent R (Sum.elim (fun x => ↑(f x)) g)
  -/
  refine .sum_type (hf.map' M'.subtype M'.ker_subtype) (.of_comp M'.mkQ hg) ?_
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    M' : Submodule R M
    ι₁ : Type u_2
    ι₂ : Type u_3
    f : ι₁ → Subtype fun x => Membership.mem M' x
    hf : LinearIndependent R f
    g : ι₂ → M
    hg : LinearIndependent R (Function.comp Submodule.Quotient.mk g)
    ⊢ Disjoint (Submodule.span R (Set.range fun x => ↑(f x))) (Submodule.span R (S …
  -/
  refine disjoint_def.mpr fun x h₁ h₂ ↦ ?_
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    M' : Submodule R M
    ι₁ : Type u_2
    ι₂ : Type u_3
    f : ι₁ → Subtype fun x => Membership.mem M' x
    hf : LinearIndependent R f
    g : ι₂ → M
    hg : LinearIndependent R (Function.comp Submodule.Quotient.mk g)
    x : M
    h₁ : Membership.mem (Submodule.span R (Set.range fun x => ↑(f x))) x
    h₂ : Membership.mem (Submodule.span R (Set.range g)) x
    ⊢ Eq x 0
  -/
  have : x ∈ M' := span_le.mpr (Set.range_subset_iff.mpr fun i ↦ (f i).prop) h₁
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    M' : Submodule R M
    ι₁ : Type u_2
    ι₂ : Type u_3
    f : ι₁ → Subtype fun x => Membership.mem M' x
    hf : LinearIndependent R f
    g : ι₂ → M
    hg : LinearIndependent R (Function.comp Submodule.Quotient.mk g)
    x : M
    h₁ : Membership.mem (Submodule.span R (Set.range fun x => ↑(f x))) x
    h₂ : Membership.mem (Submodule.span R (Set.range g)) x
    this : Membership.mem M' x
    ⊢ Eq x 0
  -/
  obtain ⟨c, rfl⟩ := Finsupp.mem_span_range_iff_exists_finsupp.mp h₂
  /-
    case intro
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    M' : Submodule R M
    ι₁ : Type u_2
    ι₂ : Type u_3
    f : ι₁ → Subtype fun x => Membership.mem M' x
    hf : LinearIndependent R f
    g : ι₂ → M
    hg : LinearIndependent R (Function.comp Submodule.Quotient.mk g)
    c : Finsupp ι₂ R
    h₁ : Membership.mem (Submodule.span R (Set.range fun x => ↑(f x))) (c.sum fun  …
    h₂ : Membership.mem (Submodule.span R (Set.range g)) (c.sum fun i a => HSMul.h …
    this : Membership.mem M' (c.sum fun i a => HSMul.hSMul a (g i))
    ⊢ Eq (c.sum fun i a => HSMul.hSMul a (g i)) 0
  -/
  simp_rw [← Quotient.mk_eq_zero, ← mkQ_apply, map_finsupp_sum, map_smul, mkQ_apply] at this
  /-
    case intro
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    M' : Submodule R M
    ι₁ : Type u_2
    ι₂ : Type u_3
    f : ι₁ → Subtype fun x => Membership.mem M' x
    hf : LinearIndependent R f
    g : ι₂ → M
    hg : LinearIndependent R (Function.comp Submodule.Quotient.mk g)
    c : Finsupp ι₂ R
    h₁ : Membership.mem (Submodule.span R (Set.range fun x => ↑(f x))) (c.sum fun  …
    h₂ : Membership.mem (Submodule.span R (Set.range g)) (c.sum fun i a => HSMul.h …
    this : Eq (c.sum fun a b => HSMul.hSMul b (Submodule.Quotient.mk (g a))) 0
    ⊢ Eq (c.sum fun i a => HSMul.hSMul a (g i)) 0
  -/
  rw [linearIndependent_iff.mp hg _ this, Finsupp.sum_zero_index]
  /-
    🎉 no goals
  -/


theorem LinearIndependent.union_of_quotient
    {M' : Submodule R M} {s : Set M} (hs : s ⊆ M') (hs' : LinearIndependent (ι := s) R Subtype.val)
  {t : Set M} (ht : LinearIndependent (ι := t) R (Submodule.Quotient.mk (p := M') ∘ Subtype.val)) :
    LinearIndependent (ι := (s ∪ t : _)) R Subtype.val := by
  refine (LinearIndependent.sum_elim_of_quotient (f := Set.embeddingOfSubset s M' hs)
    (of_comp M'.subtype (by simpa using hs')) Subtype.val ht).to_subtype_range' ?_
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    M' : Submodule R M
    s : Set M
    hs : HasSubset.Subset s ↑M'
    hs' : LinearIndependent R Subtype.val
    t : Set M
    ht : LinearIndependent R (Function.comp Submodule.Quotient.mk Subtype.val)
    ⊢ Eq (Set.range (Sum.elim (fun x => ↑((s.embeddingOfSubset (↑M') hs) x)) Subty …
  -/
  simp only [embeddingOfSubset_apply_coe, Sum.elim_range, Subtype.range_val]
  /-
    🎉 no goals
  -/


theorem rank_quotient_add_rank_le [Nontrivial R] (M' : Submodule R M) :
    Module.rank R (M ⧸ M') + Module.rank R M' ≤ Module.rank R M := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    M' : Submodule R M
    ⊢ LE.le (HAdd.hAdd (Module.rank R (HasQuotient.Quotient M M')) (Module.rank R  …
  -/
  conv_lhs => simp only [Module.rank_def]
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    M' : Submodule R M
    ⊢ LE.le (HAdd.hAdd (iSup fun ι => Cardinal.mk ↑↑ι) (iSup fun ι => Cardinal.mk  …
  -/
  have := nonempty_linearIndependent_set R (M ⧸ M')
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    M' : Submodule R M
    this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
    ⊢ LE.le (HAdd.hAdd (iSup fun ι => Cardinal.mk ↑↑ι) (iSup fun ι => Cardinal.mk  …
  -/
  have := nonempty_linearIndependent_set R M'
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    M' : Submodule R M
    this✝ : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
    this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
    ⊢ LE.le (HAdd.hAdd (iSup fun ι => Cardinal.mk ↑↑ι) (iSup fun ι => Cardinal.mk  …
  -/
  rw [Cardinal.ciSup_add_ciSup _ (bddAbove_range _) _ (bddAbove_range _)]
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    M' : Submodule R M
    this✝ : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
    this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
    ⊢ LE.le (iSup fun i => iSup fun j => HAdd.hAdd (Cardinal.mk ↑↑i) (Cardinal.mk  …
  -/
  refine ciSup_le fun ⟨s, hs⟩ ↦ ciSup_le fun ⟨t, ht⟩ ↦ ?_
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    M' : Submodule R M
    this✝ : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
    this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
    x✝¹ : Subtype fun s => LinearIndependent R Subtype.val
    s : Set (HasQuotient.Quotient M M')
    hs : LinearIndependent R Subtype.val
    x✝ : Subtype fun s => LinearIndependent R Subtype.val
    t : Set (Subtype fun x => Membership.mem M' x)
    ht : LinearIndependent R Subtype.val
    ⊢ LE.le (HAdd.hAdd (Cardinal.mk ↑↑⟨s, hs⟩) (Cardinal.mk ↑↑⟨t, ht⟩)) (Module.ra …
  -/
  choose f hf using Submodule.Quotient.mk_surjective M'
  simpa [add_comm] using (LinearIndependent.sum_elim_of_quotient ht (fun (i : s) ↦ f i)
    (by simpa [Function.comp_def, hf] using hs)).cardinal_le_rank


theorem rank_quotient_le (p : Submodule R M) : Module.rank R (M ⧸ p) ≤ Module.rank R M :=
  (mkQ p).rank_le_of_surjective Quot.mk_surjective


@[simp]
theorem rank_ulift : Module.rank R (ULift.{w} M) = Cardinal.lift.{w} (Module.rank R M) :=
  Cardinal.lift_injective.{v} <| Eq.symm <| (lift_lift _).trans ULift.moduleEquiv.symm.lift_rank_eq


@[simp]
theorem finrank_ulift : finrank R (ULift M) = finrank R M := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (Module.finrank R (ULift.{u_2, v} M)) (Module.finrank R M)
  -/
  simp_rw [finrank, rank_ulift, toNat_lift]
  /-
    🎉 no goals
  -/


open LinearMap in
theorem lift_rank_add_lift_rank_le_rank_prod [Nontrivial R] :
    lift.{v'} (Module.rank R M) + lift.{v} (Module.rank R M') ≤ Module.rank R (M × M') := by
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    inst✝ : Nontrivial R
    ⊢ LE.le (HAdd.hAdd (Cardinal.lift.{v', v} (Module.rank R M)) (Cardinal.lift.{v …
  -/
  convert rank_quotient_add_rank_le (ker <| LinearMap.fst R M M')
    /-
      case h.e'_3.h.e'_5
      R : Type u
      M : Type v
      M' : Type v'
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M
      inst✝¹ : Module R M'
      inst✝ : Nontrivial R
      ⊢ Eq (Cardinal.lift.{v', v} (Module.rank R M)) (Module.rank R (HasQuotient.Quo …
    -/
  · refine Eq.trans ?_ (lift_id'.{v, v'} _)
    rw [(quotKerEquivRange _).lift_rank_eq,
        rank_range_of_surjective _ fst_surjective, lift_umax.{v, v'}]
    /-
      case h.e'_3.h.e'_6
      R : Type u
      M : Type v
      M' : Type v'
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M
      inst✝¹ : Module R M'
      inst✝ : Nontrivial R
      ⊢ Eq (Cardinal.lift.{v, v'} (Module.rank R M')) (Module.rank R (Subtype fun x  …
    -/
  · refine Eq.trans ?_ (lift_id'.{v', v} _)
    rw [ker_fst, ← (LinearEquiv.ofInjective _ <| inr_injective (M := M) (M₂ := M')).lift_rank_eq,
        lift_umax.{v', v}]


theorem rank_add_rank_le_rank_prod [Nontrivial R] :
    Module.rank R M + Module.rank R M₁ ≤ Module.rank R (M × M₁) := by
  /-
    R : Type u
    M M₁ : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M₁
    inst✝² : Module R M
    inst✝¹ : Module R M₁
    inst✝ : Nontrivial R
    ⊢ LE.le (HAdd.hAdd (Module.rank R M) (Module.rank R M₁)) (Module.rank R (Prod  …
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  convert ← lift_rank_add_lift_rank_le_rank_prod R M M₁ <;> apply lift_id
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- If `M` and `M'` are free, then the rank of `M × M'` is
`(Module.rank R M).lift + (Module.rank R M').lift`. -/
@[simp]
theorem rank_prod : Module.rank R (M × M') =
    Cardinal.lift.{v'} (Module.rank R M) + Cardinal.lift.{v, v'} (Module.rank R M') := by
  simpa [rank_eq_card_chooseBasisIndex R M, rank_eq_card_chooseBasisIndex R M', lift_umax]
    using ((chooseBasis R M).prod (chooseBasis R M')).mk_eq_rank.symm


/-- If `M` and `M'` are free (and lie in the same universe), the rank of `M × M'` is
  `(Module.rank R M) + (Module.rank R M')`. -/
                                                                                       /-
                                                                                         R : Type u
                                                                                         M M₁ : Type v
                                                                                         inst✝⁷ : Ring R
                                                                                         inst✝⁶ : AddCommGroup M
                                                                                         inst✝⁵ : AddCommGroup M₁
                                                                                         inst✝⁴ : Module R M
                                                                                         inst✝³ : Module R M₁
                                                                                         inst✝² : StrongRankCondition R
                                                                                         inst✝¹ : Module.Free R M
                                                                                         inst✝ : Module.Free R M₁
                                                                                         ⊢ Eq (Module.rank R (Prod M M₁)) (HAdd.hAdd (Module.rank R M) (Module.rank R M …
                                                                                       -/
theorem rank_prod' : Module.rank R (M × M₁) = Module.rank R M + Module.rank R M₁ := by simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


/-- The finrank of `M × M'` is `(finrank R M) + (finrank R M')`. -/
@[simp]
theorem Module.finrank_prod [Module.Finite R M] [Module.Finite R M'] :
    finrank R (M × M') = finrank R M + finrank R M' := by
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁹ : Ring R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup M'
    inst✝⁶ : Module R M
    inst✝⁵ : Module R M'
    inst✝⁴ : StrongRankCondition R
    inst✝³ : Module.Free R M
    inst✝² : Module.Free R M'
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Finite R M'
    ⊢ Eq (Module.finrank R (Prod M M')) (HAdd.hAdd (Module.finrank R M) (Module.fi …
  -/
  simp [finrank, rank_lt_aleph0 R M, rank_lt_aleph0 R M']
  /-
    🎉 no goals
  -/


@[simp]
theorem rank_finsupp (ι : Type w) :
    Module.rank R (ι →₀ M) = Cardinal.lift.{v} #ι * Cardinal.lift.{w} (Module.rank R M) := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Free R M
    ι : Type w
    ⊢ Eq (Module.rank R (Finsupp ι M)) (HMul.hMul (Cardinal.lift.{v, w} (Cardinal. …
  -/
  obtain ⟨⟨_, bs⟩⟩ := Module.Free.exists_basis (R := R) (M := M)
  rw [← bs.mk_eq_rank'', ← (Finsupp.basis fun _ : ι => bs).mk_eq_rank'', Cardinal.mk_sigma,
    Cardinal.sum_const]


theorem rank_finsupp' (ι : Type v) : Module.rank R (ι →₀ M) = #ι * Module.rank R M := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Free R M
    ι : Type v
    ⊢ Eq (Module.rank R (Finsupp ι M)) (HMul.hMul (Cardinal.mk ι) (Module.rank R M))
  -/
  simp [rank_finsupp]
  /-
    🎉 no goals
  -/


/-- The rank of `(ι →₀ R)` is `(#ι).lift`. -/
-- Porting note, this should not be `@[simp]`, as simp can prove it.
-- @[simp]
theorem rank_finsupp_self (ι : Type w) : Module.rank R (ι →₀ R) = Cardinal.lift.{u} #ι := by
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : StrongRankCondition R
    ι : Type w
    ⊢ Eq (Module.rank R (Finsupp ι R)) (Cardinal.lift.{u, w} (Cardinal.mk ι))
  -/
  simp [rank_finsupp]
  /-
    🎉 no goals
  -/


/-- If `R` and `ι` lie in the same universe, the rank of `(ι →₀ R)` is `# ι`. -/
                                                                            /-
                                                                              R : Type u
                                                                              inst✝¹ : Ring R
                                                                              inst✝ : StrongRankCondition R
                                                                              ι : Type u
                                                                              ⊢ Eq (Module.rank R (Finsupp ι R)) (Cardinal.mk ι)
                                                                            -/
theorem rank_finsupp_self' {ι : Type u} : Module.rank R (ι →₀ R) = #ι := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- The rank of the direct sum is the sum of the ranks. -/
@[simp]
theorem rank_directSum {ι : Type v} (M : ι → Type w) [∀ i : ι, AddCommGroup (M i)]
    [∀ i : ι, Module R (M i)] [∀ i : ι, Module.Free R (M i)] :
    Module.rank R (⨁ i, M i) = Cardinal.sum fun i => Module.rank R (M i) := by
  /-
    R : Type u
    inst✝⁴ : Ring R
    inst✝³ : StrongRankCondition R
    ι : Type v
    M : ι → Type w
    inst✝² : (i : ι) → AddCommGroup (M i)
    inst✝¹ : (i : ι) → Module R (M i)
    inst✝ : ∀ (i : ι), Module.Free R (M i)
    ⊢ Eq (Module.rank R (DirectSum ι fun i => M i)) (Cardinal.sum fun i => Module. …
  -/
  let B i := chooseBasis R (M i)
  /-
    R : Type u
    inst✝⁴ : Ring R
    inst✝³ : StrongRankCondition R
    ι : Type v
    M : ι → Type w
    inst✝² : (i : ι) → AddCommGroup (M i)
    inst✝¹ : (i : ι) → Module R (M i)
    inst✝ : ∀ (i : ι), Module.Free R (M i)
    B : (i : ι) → Basis (Module.Free.ChooseBasisIndex R (M i)) R (M i) := fun i => …
    ⊢ Eq (Module.rank R (DirectSum ι fun i => M i)) (Cardinal.sum fun i => Module. …
  -/
  let b : Basis _ R (⨁ i, M i) := DFinsupp.basis fun i => B i
  /-
    R : Type u
    inst✝⁴ : Ring R
    inst✝³ : StrongRankCondition R
    ι : Type v
    M : ι → Type w
    inst✝² : (i : ι) → AddCommGroup (M i)
    inst✝¹ : (i : ι) → Module R (M i)
    inst✝ : ∀ (i : ι), Module.Free R (M i)
    B : (i : ι) → Basis (Module.Free.ChooseBasisIndex R (M i)) R (M i) := fun i => …
    b : Basis (Sigma fun i => Module.Free.ChooseBasisIndex R (M i)) R (DirectSum ι …
    ⊢ Eq (Module.rank R (DirectSum ι fun i => M i)) (Cardinal.sum fun i => Module. …
  -/
  simp [← b.mk_eq_rank'', fun i => (B i).mk_eq_rank'']
  /-
    🎉 no goals
  -/


/-- If `m` and `n` are finite, the rank of `m × n` matrices over a module `M` is
`(#m).lift * (#n).lift * rank R M`. -/
@[simp]
theorem rank_matrix_module (m : Type w) (n : Type w') [Finite m] [Finite n] :
    Module.rank R (Matrix m n M) =
      lift.{max v w'} #m * lift.{max v w} #n * lift.{max w w'} (Module.rank R M) := by
  /-
    R : Type u
    M : Type v
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : StrongRankCondition R
    inst✝² : Module.Free R M
    m : Type w
    n : Type w'
    inst✝¹ : Finite m
    inst✝ : Finite n
    ⊢ Eq (Module.rank R (Matrix m n M)) (HMul.hMul (HMul.hMul (Cardinal.lift.{max  …
  -/
  cases nonempty_fintype m
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : StrongRankCondition R
    inst✝² : Module.Free R M
    m : Type w
    n : Type w'
    inst✝¹ : Finite m
    inst✝ : Finite n
    val✝ : Fintype m
    ⊢ Eq (Module.rank R (Matrix m n M)) (HMul.hMul (HMul.hMul (Cardinal.lift.{max  …
  -/
  cases nonempty_fintype n
  /-
    case intro.intro
    R : Type u
    M : Type v
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : StrongRankCondition R
    inst✝² : Module.Free R M
    m : Type w
    n : Type w'
    inst✝¹ : Finite m
    inst✝ : Finite n
    val✝¹ : Fintype m
    val✝ : Fintype n
    ⊢ Eq (Module.rank R (Matrix m n M)) (HMul.hMul (HMul.hMul (Cardinal.lift.{max  …
  -/
  obtain ⟨I, b⟩ := Module.Free.exists_basis (R := R) (M := M)
  /-
    case intro.intro.intro.mk
    R : Type u
    M : Type v
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : StrongRankCondition R
    inst✝² : Module.Free R M
    m : Type w
    n : Type w'
    inst✝¹ : Finite m
    inst✝ : Finite n
    val✝¹ : Fintype m
    val✝ : Fintype n
    I : Type v
    b : Basis I R M
    ⊢ Eq (Module.rank R (Matrix m n M)) (HMul.hMul (HMul.hMul (Cardinal.lift.{max  …
  -/
  rw [← (b.matrix m n).mk_eq_rank'']
  /-
    case intro.intro.intro.mk
    R : Type u
    M : Type v
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : StrongRankCondition R
    inst✝² : Module.Free R M
    m : Type w
    n : Type w'
    inst✝¹ : Finite m
    inst✝ : Finite n
    val✝¹ : Fintype m
    val✝ : Fintype n
    I : Type v
    b : Basis I R M
    ⊢ Eq (Cardinal.mk (Prod m (Prod n I))) (HMul.hMul (HMul.hMul (Cardinal.lift.{m …
  -/
  simp only [mk_prod, lift_mul, lift_lift, ← mul_assoc, b.mk_eq_rank'']
  /-
    🎉 no goals
  -/



/-- If `m` and `n` are finite and lie in the same universe, the rank of `m × n` matrices over a
module `M` is `(#m * #n).lift * rank R M`. -/
@[simp high]
theorem rank_matrix_module' (m n : Type w) [Finite m] [Finite n] :
    Module.rank R (Matrix m n M) =
      lift.{max v} (#m * #n) * lift.{w} (Module.rank R M) := by
  /-
    R : Type u
    M : Type v
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : StrongRankCondition R
    inst✝² : Module.Free R M
    m n : Type w
    inst✝¹ : Finite m
    inst✝ : Finite n
    ⊢ Eq (Module.rank R (Matrix m n M)) (HMul.hMul (Cardinal.lift.{v, w} (HMul.hMu …
  -/
  rw [rank_matrix_module, lift_mul, lift_umax.{w, v}]
  /-
    🎉 no goals
  -/


/-- If `m` and `n` are finite, the rank of `m × n` matrices is `(#m).lift * (#n).lift`. -/
theorem rank_matrix (m : Type v) (n : Type w) [Finite m] [Finite n] :
    Module.rank R (Matrix m n R) =
      Cardinal.lift.{max v w u, v} #m * Cardinal.lift.{max v w u, w} #n := by
  rw [rank_matrix_module, rank_self, lift_one, mul_one, ← lift_lift.{v, max u w}, lift_id,
    ← lift_lift.{w, max u v}, lift_id]


/-- If `m` and `n` are finite and lie in the same universe, the rank of `m × n` matrices is
  `(#n * #m).lift`. -/
theorem rank_matrix' (m n : Type v) [Finite m] [Finite n] :
    Module.rank R (Matrix m n R) = Cardinal.lift.{u} (#m * #n) := by
  /-
    R : Type u
    inst✝³ : Ring R
    inst✝² : StrongRankCondition R
    m n : Type v
    inst✝¹ : Finite m
    inst✝ : Finite n
    ⊢ Eq (Module.rank R (Matrix m n R)) (Cardinal.lift.{u, v} (HMul.hMul (Cardinal …
  -/
  rw [rank_matrix, lift_mul, lift_umax.{v, u}]
  /-
    🎉 no goals
  -/


/-- If `m` and `n` are finite and lie in the same universe as `R`, the rank of `m × n` matrices
  is `# m * # n`. -/
theorem rank_matrix'' (m n : Type u) [Finite m] [Finite n] :
                                                 /-
                                                   R : Type u
                                                   inst✝³ : Ring R
                                                   inst✝² : StrongRankCondition R
                                                   m n : Type u
                                                   inst✝¹ : Finite m
                                                   inst✝ : Finite n
                                                   ⊢ Eq (Module.rank R (Matrix m n R)) (HMul.hMul (Cardinal.mk m) (Cardinal.mk n))
                                                 -/
    Module.rank R (Matrix m n R) = #m * #n := by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem finrank_finsupp {ι : Type v} [Fintype ι] : finrank R (ι →₀ M) = card ι * finrank R M := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    inst✝¹ : Module.Free R M
    ι : Type v
    inst✝ : Fintype ι
    ⊢ Eq (Module.finrank R (Finsupp ι M)) (HMul.hMul (Fintype.card ι) (Module.finr …
  -/
  rw [finrank, finrank, rank_finsupp, ← mk_toNat_eq_card, toNat_mul, toNat_lift, toNat_lift]
  /-
    🎉 no goals
  -/


/-- The finrank of `(ι →₀ R)` is `Fintype.card ι`. -/
@[simp]
theorem finrank_finsupp_self {ι : Type v} [Fintype ι] : finrank R (ι →₀ R) = card ι := by
  /-
    R : Type u
    inst✝² : Ring R
    inst✝¹ : StrongRankCondition R
    ι : Type v
    inst✝ : Fintype ι
    ⊢ Eq (Module.finrank R (Finsupp ι R)) (Fintype.card ι)
  -/
  rw [finrank, rank_finsupp_self, ← mk_toNat_eq_card, toNat_lift]
  /-
    🎉 no goals
  -/


/-- The finrank of the direct sum is the sum of the finranks. -/
@[simp]
theorem finrank_directSum {ι : Type v} [Fintype ι] (M : ι → Type w) [∀ i : ι, AddCommGroup (M i)]
    [∀ i : ι, Module R (M i)] [∀ i : ι, Module.Free R (M i)] [∀ i : ι, Module.Finite R (M i)] :
    finrank R (⨁ i, M i) = ∑ i, finrank R (M i) := by
  /-
    R : Type u
    inst✝⁶ : Ring R
    inst✝⁵ : StrongRankCondition R
    ι : Type v
    inst✝⁴ : Fintype ι
    M : ι → Type w
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : ∀ (i : ι), Module.Free R (M i)
    inst✝ : ∀ (i : ι), Module.Finite R (M i)
    ⊢ Eq (Module.finrank R (DirectSum ι fun i => M i)) (Finset.univ.sum fun i => M …
  -/
  letI := nontrivial_of_invariantBasisNumber R
  simp only [finrank, fun i => rank_eq_card_chooseBasisIndex R (M i), rank_directSum, ← mk_sigma,
    mk_toNat_eq_card, card_sigma]


/-- If `m` and `n` are `Fintype`, the finrank of `m × n` matrices over a module `M` is
  `(Fintype.card m) * (Fintype.card n) * finrank R M`. -/
theorem finrank_matrix (m n : Type*) [Fintype m] [Fintype n] :
                                                                   /-
                                                                     R : Type u
                                                                     M : Type v
                                                                     inst✝⁶ : Ring R
                                                                     inst✝⁵ : AddCommGroup M
                                                                     inst✝⁴ : Module R M
                                                                     inst✝³ : StrongRankCondition R
                                                                     inst✝² : Module.Free R M
                                                                     m : Type u_2
                                                                     n : Type u_3
                                                                     inst✝¹ : Fintype m
                                                                     inst✝ : Fintype n
                                                                     ⊢ Eq (Module.finrank R (Matrix m n M)) (HMul.hMul (HMul.hMul (Fintype.card m)  …
                                                                   -/
    finrank R (Matrix m n M) = card m * card n * finrank R M := by simp [finrank]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- The rank of a finite product of free modules is the sum of the ranks. -/
-- this result is not true without the freeness assumption
@[simp]
theorem rank_pi [Finite η] : Module.rank R (∀ i, φ i) =
    Cardinal.sum fun i => Module.rank R (φ i) := by
  /-
    R : Type u
    η : Type u₁'
    φ : η → Type u_1
    inst✝⁵ : Ring R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : (i : η) → AddCommGroup (φ i)
    inst✝² : (i : η) → Module R (φ i)
    inst✝¹ : ∀ (i : η), Module.Free R (φ i)
    inst✝ : Finite η
    ⊢ Eq (Module.rank R ((i : η) → φ i)) (Cardinal.sum fun i => Module.rank R (φ i))
  -/
  cases nonempty_fintype η
  /-
    case intro
    R : Type u
    η : Type u₁'
    φ : η → Type u_1
    inst✝⁵ : Ring R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : (i : η) → AddCommGroup (φ i)
    inst✝² : (i : η) → Module R (φ i)
    inst✝¹ : ∀ (i : η), Module.Free R (φ i)
    inst✝ : Finite η
    val✝ : Fintype η
    ⊢ Eq (Module.rank R ((i : η) → φ i)) (Cardinal.sum fun i => Module.rank R (φ i))
  -/
  let B i := chooseBasis R (φ i)
  /-
    case intro
    R : Type u
    η : Type u₁'
    φ : η → Type u_1
    inst✝⁵ : Ring R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : (i : η) → AddCommGroup (φ i)
    inst✝² : (i : η) → Module R (φ i)
    inst✝¹ : ∀ (i : η), Module.Free R (φ i)
    inst✝ : Finite η
    val✝ : Fintype η
    B : (i : η) → Basis (Module.Free.ChooseBasisIndex R (φ i)) R (φ i) := fun i => …
    ⊢ Eq (Module.rank R ((i : η) → φ i)) (Cardinal.sum fun i => Module.rank R (φ i))
  -/
  let b : Basis _ R (∀ i, φ i) := Pi.basis fun i => B i
  /-
    case intro
    R : Type u
    η : Type u₁'
    φ : η → Type u_1
    inst✝⁵ : Ring R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : (i : η) → AddCommGroup (φ i)
    inst✝² : (i : η) → Module R (φ i)
    inst✝¹ : ∀ (i : η), Module.Free R (φ i)
    inst✝ : Finite η
    val✝ : Fintype η
    B : (i : η) → Basis (Module.Free.ChooseBasisIndex R (φ i)) R (φ i) := fun i => …
    b : Basis (Sigma fun j => Module.Free.ChooseBasisIndex R (φ j)) R ((i : η) → φ …
    ⊢ Eq (Module.rank R ((i : η) → φ i)) (Cardinal.sum fun i => Module.rank R (φ i))
  -/
  simp [← b.mk_eq_rank'', fun i => (B i).mk_eq_rank'']
  /-
    🎉 no goals
  -/


/-- The finrank of `(ι → R)` is `Fintype.card ι`. -/
theorem Module.finrank_pi {ι : Type v} [Fintype ι] :
    finrank R (ι → R) = Fintype.card ι := by
  /-
    R : Type u
    inst✝² : Ring R
    inst✝¹ : StrongRankCondition R
    ι : Type v
    inst✝ : Fintype ι
    ⊢ Eq (Module.finrank R (ι → R)) (Fintype.card ι)
  -/
  simp [finrank]
  /-
    🎉 no goals
  -/

--TODO: this should follow from `LinearEquiv.finrank_eq`, that is over a field.

/-- The finrank of a finite product is the sum of the finranks. -/
theorem Module.finrank_pi_fintype
    {ι : Type v} [Fintype ι] {M : ι → Type w} [∀ i : ι, AddCommGroup (M i)]
    [∀ i : ι, Module R (M i)] [∀ i : ι, Module.Free R (M i)] [∀ i : ι, Module.Finite R (M i)] :
    finrank R (∀ i, M i) = ∑ i, finrank R (M i) := by
  /-
    R : Type u
    inst✝⁶ : Ring R
    inst✝⁵ : StrongRankCondition R
    ι : Type v
    inst✝⁴ : Fintype ι
    M : ι → Type w
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : ∀ (i : ι), Module.Free R (M i)
    inst✝ : ∀ (i : ι), Module.Finite R (M i)
    ⊢ Eq (Module.finrank R ((i : ι) → M i)) (Finset.univ.sum fun i => Module.finra …
  -/
  letI := nontrivial_of_invariantBasisNumber R
  simp only [finrank, fun i => rank_eq_card_chooseBasisIndex R (M i), rank_pi, ← mk_sigma,
    mk_toNat_eq_card, Fintype.card_sigma]


theorem rank_fun {M η : Type u} [Fintype η] [AddCommGroup M] [Module R M] [Module.Free R M] :
    Module.rank R (η → M) = Fintype.card η * Module.rank R M := by
  /-
    R : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : StrongRankCondition R
    M η : Type u
    inst✝³ : Fintype η
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    ⊢ Eq (Module.rank R (η → M)) (HMul.hMul (↑(Fintype.card η)) (Module.rank R M))
  -/
  rw [rank_pi, Cardinal.sum_const', Cardinal.mk_fintype]
  /-
    🎉 no goals
  -/


theorem rank_fun_eq_lift_mul : Module.rank R (η → M) =
    (Fintype.card η : Cardinal.{max u₁' v}) * Cardinal.lift.{u₁'} (Module.rank R M) := by
  /-
    R : Type u
    M : Type v
    η : Type u₁'
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : StrongRankCondition R
    inst✝¹ : Module.Free R M
    inst✝ : Fintype η
    ⊢ Eq (Module.rank R (η → M)) (HMul.hMul (↑(Fintype.card η)) (Cardinal.lift.{u₁ …
  -/
  rw [rank_pi, Cardinal.sum_const, Cardinal.mk_fintype, Cardinal.lift_natCast]
  /-
    🎉 no goals
  -/


theorem rank_fun' : Module.rank R (η → R) = Fintype.card η := by
  /-
    R : Type u
    η : Type u₁'
    inst✝² : Ring R
    inst✝¹ : StrongRankCondition R
    inst✝ : Fintype η
    ⊢ Eq (Module.rank R (η → R)) ↑(Fintype.card η)
  -/
  rw [rank_fun_eq_lift_mul, rank_self, Cardinal.lift_one, mul_one]
  /-
    🎉 no goals
  -/


                                                                   /-
                                                                     R : Type u
                                                                     inst✝¹ : Ring R
                                                                     inst✝ : StrongRankCondition R
                                                                     n : Nat
                                                                     ⊢ Eq (Module.rank R (Fin n → R)) ↑n
                                                                   -/
theorem rank_fin_fun (n : ℕ) : Module.rank R (Fin n → R) = n := by simp [rank_fun']
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- The vector space of functions on a `Fintype ι` has finrank equal to the cardinality of `ι`. -/
@[simp]
theorem Module.finrank_fintype_fun_eq_card : finrank R (η → R) = Fintype.card η :=
  finrank_eq_of_rank_eq rank_fun'


/-- The vector space of functions on `Fin n` has finrank equal to `n`. -/
                                                                         /-
                                                                           R : Type u
                                                                           inst✝¹ : Ring R
                                                                           inst✝ : StrongRankCondition R
                                                                           n : Nat
                                                                           ⊢ Eq (Module.finrank R (Fin n → R)) n
                                                                         -/
theorem Module.finrank_fin_fun {n : ℕ} : finrank R (Fin n → R) = n := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- An `n`-dimensional `R`-vector space is equivalent to `Fin n → R`. -/
def finDimVectorspaceEquiv (n : ℕ) (hn : Module.rank R M = n) : M ≃ₗ[R] Fin n → R := by
  /-
    R : Type u
    S : Type u'
    M : Type v
    M' : Type v'
    M₁ : Type v
    ι : Type w
    ι' : Type w'
    η : Type u₁'
    φ : η → Type u_1
    inst✝¹¹ : Ring R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : AddCommGroup M'
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M
    inst✝⁵ : StrongRankCondition R
    inst✝⁴ : Module.Free R M
    inst✝³ : (i : η) → AddCommGroup (φ i)
    inst✝² : (i : η) → Module R (φ i)
    inst✝¹ : ∀ (i : η), Module.Free R (φ i)
    inst✝ : Fintype η
    n : Nat
    hn : Eq (Module.rank R M) ↑n
    ⊢ LinearEquiv (RingHom.id R) M (Fin n → R)
  -/
  haveI := nontrivial_of_invariantBasisNumber R
  /-
    R : Type u
    S : Type u'
    M : Type v
    M' : Type v'
    M₁ : Type v
    ι : Type w
    ι' : Type w'
    η : Type u₁'
    φ : η → Type u_1
    inst✝¹¹ : Ring R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : AddCommGroup M'
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M
    inst✝⁵ : StrongRankCondition R
    inst✝⁴ : Module.Free R M
    inst✝³ : (i : η) → AddCommGroup (φ i)
    inst✝² : (i : η) → Module R (φ i)
    inst✝¹ : ∀ (i : η), Module.Free R (φ i)
    inst✝ : Fintype η
    n : Nat
    hn : Eq (Module.rank R M) ↑n
    this : Nontrivial R
    ⊢ LinearEquiv (RingHom.id R) M (Fin n → R)
  -/
  have : Cardinal.lift.{u} (n : Cardinal.{v}) = Cardinal.lift.{v} (n : Cardinal.{u}) := by simp
  /-
    R : Type u
    S : Type u'
    M : Type v
    M' : Type v'
    M₁ : Type v
    ι : Type w
    ι' : Type w'
    η : Type u₁'
    φ : η → Type u_1
    inst✝¹¹ : Ring R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : AddCommGroup M'
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M
    inst✝⁵ : StrongRankCondition R
    inst✝⁴ : Module.Free R M
    inst✝³ : (i : η) → AddCommGroup (φ i)
    inst✝² : (i : η) → Module R (φ i)
    inst✝¹ : ∀ (i : η), Module.Free R (φ i)
    inst✝ : Fintype η
    n : Nat
    hn : Eq (Module.rank R M) ↑n
    this✝ : Nontrivial R
    this : Eq (Cardinal.lift.{u, v} ↑n) (Cardinal.lift.{v, u} ↑n)
    ⊢ LinearEquiv (RingHom.id R) M (Fin n → R)
  -/
  have hn := Cardinal.lift_inj.{v, u}.2 hn
  /-
    R : Type u
    S : Type u'
    M : Type v
    M' : Type v'
    M₁ : Type v
    ι : Type w
    ι' : Type w'
    η : Type u₁'
    φ : η → Type u_1
    inst✝¹¹ : Ring R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : AddCommGroup M'
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M
    inst✝⁵ : StrongRankCondition R
    inst✝⁴ : Module.Free R M
    inst✝³ : (i : η) → AddCommGroup (φ i)
    inst✝² : (i : η) → Module R (φ i)
    inst✝¹ : ∀ (i : η), Module.Free R (φ i)
    inst✝ : Fintype η
    n : Nat
    hn✝ : Eq (Module.rank R M) ↑n
    this✝ : Nontrivial R
    this : Eq (Cardinal.lift.{u, v} ↑n) (Cardinal.lift.{v, u} ↑n)
    hn : Eq (Cardinal.lift.{u, v} (Module.rank R M)) (Cardinal.lift.{u, v} ↑n)
    ⊢ LinearEquiv (RingHom.id R) M (Fin n → R)
  -/
  rw [this] at hn
  /-
    R : Type u
    S : Type u'
    M : Type v
    M' : Type v'
    M₁ : Type v
    ι : Type w
    ι' : Type w'
    η : Type u₁'
    φ : η → Type u_1
    inst✝¹¹ : Ring R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : AddCommGroup M'
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M
    inst✝⁵ : StrongRankCondition R
    inst✝⁴ : Module.Free R M
    inst✝³ : (i : η) → AddCommGroup (φ i)
    inst✝² : (i : η) → Module R (φ i)
    inst✝¹ : ∀ (i : η), Module.Free R (φ i)
    inst✝ : Fintype η
    n : Nat
    hn✝ : Eq (Module.rank R M) ↑n
    this✝ : Nontrivial R
    this : Eq (Cardinal.lift.{u, v} ↑n) (Cardinal.lift.{v, u} ↑n)
    hn : Eq (Cardinal.lift.{u, v} (Module.rank R M)) (Cardinal.lift.{v, u} ↑n)
    ⊢ LinearEquiv (RingHom.id R) M (Fin n → R)
  -/
  rw [← @rank_fin_fun R _ _ n] at hn
  /-
    R : Type u
    S : Type u'
    M : Type v
    M' : Type v'
    M₁ : Type v
    ι : Type w
    ι' : Type w'
    η : Type u₁'
    φ : η → Type u_1
    inst✝¹¹ : Ring R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : AddCommGroup M'
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M
    inst✝⁵ : StrongRankCondition R
    inst✝⁴ : Module.Free R M
    inst✝³ : (i : η) → AddCommGroup (φ i)
    inst✝² : (i : η) → Module R (φ i)
    inst✝¹ : ∀ (i : η), Module.Free R (φ i)
    inst✝ : Fintype η
    n : Nat
    hn✝ : Eq (Module.rank R M) ↑n
    this✝ : Nontrivial R
    this : Eq (Cardinal.lift.{u, v} ↑n) (Cardinal.lift.{v, u} ↑n)
    hn : Eq (Cardinal.lift.{u, v} (Module.rank R M)) (Cardinal.lift.{v, u} (Module …
    ⊢ LinearEquiv (RingHom.id R) M (Fin n → R)
  -/
  haveI : Module.Free R (Fin n → R) := Module.Free.pi _ _
  /-
    R : Type u
    S : Type u'
    M : Type v
    M' : Type v'
    M₁ : Type v
    ι : Type w
    ι' : Type w'
    η : Type u₁'
    φ : η → Type u_1
    inst✝¹¹ : Ring R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : AddCommGroup M'
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M
    inst✝⁵ : StrongRankCondition R
    inst✝⁴ : Module.Free R M
    inst✝³ : (i : η) → AddCommGroup (φ i)
    inst✝² : (i : η) → Module R (φ i)
    inst✝¹ : ∀ (i : η), Module.Free R (φ i)
    inst✝ : Fintype η
    n : Nat
    hn✝ : Eq (Module.rank R M) ↑n
    this✝¹ : Nontrivial R
    this✝ : Eq (Cardinal.lift.{u, v} ↑n) (Cardinal.lift.{v, u} ↑n)
    hn : Eq (Cardinal.lift.{u, v} (Module.rank R M)) (Cardinal.lift.{v, u} (Module …
    this : Module.Free R (Fin n → R)
    ⊢ LinearEquiv (RingHom.id R) M (Fin n → R)
  -/
  exact Classical.choice (nonempty_linearEquiv_of_lift_rank_eq hn)
  /-
    🎉 no goals
  -/


/-- The `S`-rank of `M ⊗[R] M'` is `(Module.rank S M).lift * (Module.rank R M').lift`. -/
@[simp]
theorem rank_tensorProduct :
    Module.rank R (M ⊗[S] M') =
      Cardinal.lift.{v'} (Module.rank R M) * Cardinal.lift.{v} (Module.rank S M') := by
  /-
    R : Type u
    S : Type u'
    M : Type v
    M' : Type v'
    inst✝¹² : Ring R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : AddCommGroup M'
    inst✝⁸ : Module R M
    inst✝⁷ : StrongRankCondition R
    inst✝⁶ : StrongRankCondition S
    inst✝⁵ : Module S M
    inst✝⁴ : Module S M'
    inst✝³ : Module.Free S M'
    inst✝² : Algebra S R
    inst✝¹ : IsScalarTower S R M
    inst✝ : Module.Free R M
    ⊢ Eq (Module.rank R (TensorProduct S M M')) (HMul.hMul (Cardinal.lift.{v', v}  …
  -/
  obtain ⟨⟨_, bM⟩⟩ := Module.Free.exists_basis (R := R) (M := M)
  /-
    case intro.mk
    R : Type u
    S : Type u'
    M : Type v
    M' : Type v'
    inst✝¹² : Ring R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : AddCommGroup M'
    inst✝⁸ : Module R M
    inst✝⁷ : StrongRankCondition R
    inst✝⁶ : StrongRankCondition S
    inst✝⁵ : Module S M
    inst✝⁴ : Module S M'
    inst✝³ : Module.Free S M'
    inst✝² : Algebra S R
    inst✝¹ : IsScalarTower S R M
    inst✝ : Module.Free R M
    fst✝ : Type v
    bM : Basis fst✝ R M
    ⊢ Eq (Module.rank R (TensorProduct S M M')) (HMul.hMul (Cardinal.lift.{v', v}  …
  -/
  obtain ⟨⟨_, bN⟩⟩ := Module.Free.exists_basis (R := S) (M := M')
  /-
    case intro.mk.intro.mk
    R : Type u
    S : Type u'
    M : Type v
    M' : Type v'
    inst✝¹² : Ring R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : AddCommGroup M'
    inst✝⁸ : Module R M
    inst✝⁷ : StrongRankCondition R
    inst✝⁶ : StrongRankCondition S
    inst✝⁵ : Module S M
    inst✝⁴ : Module S M'
    inst✝³ : Module.Free S M'
    inst✝² : Algebra S R
    inst✝¹ : IsScalarTower S R M
    inst✝ : Module.Free R M
    fst✝¹ : Type v
    bM : Basis fst✝¹ R M
    fst✝ : Type v'
    bN : Basis fst✝ S M'
    ⊢ Eq (Module.rank R (TensorProduct S M M')) (HMul.hMul (Cardinal.lift.{v', v}  …
  -/
  rw [← bM.mk_eq_rank'', ← bN.mk_eq_rank'', ← (bM.tensorProduct bN).mk_eq_rank'', Cardinal.mk_prod]
  /-
    🎉 no goals
  -/


/-- If `M` and `M'` lie in the same universe, the `S`-rank of `M ⊗[R] M'` is
  `(Module.rank S M) * (Module.rank R M')`. -/
theorem rank_tensorProduct' :
                                                                         /-
                                                                           R : Type u
                                                                           S : Type u'
                                                                           M M₁ : Type v
                                                                           inst✝¹² : Ring R
                                                                           inst✝¹¹ : CommRing S
                                                                           inst✝¹⁰ : AddCommGroup M
                                                                           inst✝⁹ : AddCommGroup M₁
                                                                           inst✝⁸ : Module R M
                                                                           inst✝⁷ : StrongRankCondition R
                                                                           inst✝⁶ : StrongRankCondition S
                                                                           inst✝⁵ : Module S M
                                                                           inst✝⁴ : Module S M₁
                                                                           inst✝³ : Module.Free S M₁
                                                                           inst✝² : Algebra S R
                                                                           inst✝¹ : IsScalarTower S R M
                                                                           inst✝ : Module.Free R M
                                                                           ⊢ Eq (Module.rank R (TensorProduct S M M₁)) (HMul.hMul (Module.rank R M) (Modu …
                                                                         -/
    Module.rank R (M ⊗[S] M₁) = Module.rank R M * Module.rank S M₁ := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem Module.rank_baseChange :
                                                                           /-
                                                                             R : Type u
                                                                             S : Type u'
                                                                             M' : Type v'
                                                                             inst✝⁷ : Ring R
                                                                             inst✝⁶ : CommRing S
                                                                             inst✝⁵ : AddCommGroup M'
                                                                             inst✝⁴ : StrongRankCondition R
                                                                             inst✝³ : StrongRankCondition S
                                                                             inst✝² : Module S M'
                                                                             inst✝¹ : Module.Free S M'
                                                                             inst✝ : Algebra S R
                                                                             ⊢ Eq (Module.rank R (TensorProduct S R M')) (Cardinal.lift.{u, v'} (Module.ran …
                                                                           -/
    Module.rank R (R ⊗[S] M') = Cardinal.lift.{u} (Module.rank S M') := by simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- The `S`-finrank of `M ⊗[R] M'` is `(finrank S M) * (finrank R M')`. -/
@[simp]
theorem Module.finrank_tensorProduct :
                                                             /-
                                                               R : Type u
                                                               S : Type u'
                                                               M : Type v
                                                               M' : Type v'
                                                               inst✝¹² : Ring R
                                                               inst✝¹¹ : CommRing S
                                                               inst✝¹⁰ : AddCommGroup M
                                                               inst✝⁹ : AddCommGroup M'
                                                               inst✝⁸ : Module R M
                                                               inst✝⁷ : StrongRankCondition R
                                                               inst✝⁶ : StrongRankCondition S
                                                               inst✝⁵ : Module S M
                                                               inst✝⁴ : Module S M'
                                                               inst✝³ : Module.Free S M'
                                                               inst✝² : Algebra S R
                                                               inst✝¹ : IsScalarTower S R M
                                                               inst✝ : Module.Free R M
                                                               ⊢ Eq (Module.finrank R (TensorProduct S M M')) (HMul.hMul (Module.finrank R M) …
                                                             -/
    finrank R (M ⊗[S] M') = finrank R M * finrank S M' := by simp [finrank]
                                                             /-
                                                               🎉 no goals
                                                             -/


                                                                               /-
                                                                                 R : Type u
                                                                                 S : Type u'
                                                                                 M' : Type v'
                                                                                 inst✝⁷ : Ring R
                                                                                 inst✝⁶ : CommRing S
                                                                                 inst✝⁵ : AddCommGroup M'
                                                                                 inst✝⁴ : StrongRankCondition R
                                                                                 inst✝³ : StrongRankCondition S
                                                                                 inst✝² : Module S M'
                                                                                 inst✝¹ : Module.Free S M'
                                                                                 inst✝ : Algebra S R
                                                                                 ⊢ Eq (Module.finrank R (TensorProduct S R M')) (Module.finrank S M')
                                                                               -/
theorem Module.finrank_baseChange : finrank R (R ⊗[S] M') = finrank S M' := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem lt_of_le_of_finrank_lt_finrank {s t : Submodule R M} (le : s ≤ t)
    (lt : finrank R s < finrank R t) : s < t :=
                                             /-
                                               R : Type u
                                               M : Type v
                                               inst✝² : Ring R
                                               inst✝¹ : AddCommGroup M
                                               inst✝ : Module R M
                                               s t : Submodule R M
                                               le : LE.le s t
                                               lt : LT.lt (Module.finrank R (Subtype fun x => Membership.mem s x)) (Module.fi …
                                               h : Eq s t
                                               ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem s x)) (Module.finrank  …
                                             -/
  lt_of_le_of_ne le fun h => ne_of_lt lt (by rw [h])
                                             /-
                                               🎉 no goals
                                             -/


theorem lt_top_of_finrank_lt_finrank {s : Submodule R M} (lt : finrank R s < finrank R M) :
    s < ⊤ := by
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Submodule R M
    lt : LT.lt (Module.finrank R (Subtype fun x => Membership.mem s x)) (Module.fi …
    ⊢ LT.lt s Top.top
  -/
  rw [← finrank_top R M] at lt
  /-
    R : Type u
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Submodule R M
    lt : LT.lt (Module.finrank R (Subtype fun x => Membership.mem s x)) (Module.fi …
    ⊢ LT.lt s Top.top
  -/
  exact lt_of_le_of_finrank_lt_finrank le_top lt
  /-
    🎉 no goals
  -/


/-- The dimension of a submodule is bounded by the dimension of the ambient space. -/
theorem Submodule.finrank_le [Module.Finite R M] (s : Submodule R M) :
    finrank R s ≤ finrank R M :=
  toNat_le_toNat (Submodule.rank_le s) (rank_lt_aleph0 _ _)


/-- The dimension of a quotient is bounded by the dimension of the ambient space. -/
theorem Submodule.finrank_quotient_le [Module.Finite R M] (s : Submodule R M) :
    finrank R (M ⧸ s) ≤ finrank R M :=
  toNat_le_toNat ((Submodule.mkQ s).rank_le_of_surjective Quot.mk_surjective)
    (rank_lt_aleph0 _ _)


/-- Pushforwards of finite submodules have a smaller finrank. -/
theorem Submodule.finrank_map_le
    [Module R M'] (f : M →ₗ[R] M') (p : Submodule R M) [Module.Finite R p] :
    finrank R (p.map f) ≤ finrank R p :=
  finrank_le_finrank_of_rank_le_rank (lift_rank_map_le _ _) (rank_lt_aleph0 _ _)


theorem Submodule.finrank_mono {s t : Submodule R M} [Module.Finite R t] (hst : s ≤ t) :
    finrank R s ≤ finrank R t :=
  Cardinal.toNat_le_toNat (Submodule.rank_mono hst) (rank_lt_aleph0 R ↥t)


@[deprecated (since := "2024-09-30")]
alias Submodule.finrank_le_finrank_of_le := Submodule.finrank_mono


theorem rank_span_le (s : Set M) : Module.rank R (span R s) ≤ #s := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    s : Set M
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (Submodule.span R s) x …
  -/
  rw [Finsupp.span_eq_range_linearCombination, ← lift_strictMono.le_iff_le]
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    s : Set M
    ⊢ LE.le (Cardinal.lift.{?u.442780, v} (Module.rank R (Subtype fun x => Members …
  -/
  refine (lift_rank_range_le _).trans ?_
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    s : Set M
    ⊢ LE.le (Cardinal.lift.{v, max u v} (Module.rank R (Finsupp (Subtype fun x =>  …
  -/
  rw [rank_finsupp_self]
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    s : Set M
    ⊢ LE.le (Cardinal.lift.{v, max u v} (Cardinal.lift.{u, v} (Cardinal.mk (Subtyp …
  -/
  simp only [lift_lift, le_refl]
  /-
    🎉 no goals
  -/


theorem rank_span_finset_le (s : Finset M) : Module.rank R (span R (s : Set M)) ≤ s.card := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    s : Finset M
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (Submodule.span R ↑s)  …
  -/
  simpa using rank_span_le s.toSet
  /-
    🎉 no goals
  -/


theorem rank_span_of_finset (s : Finset M) : Module.rank R (span R (s : Set M)) < ℵ₀ :=
  (rank_span_finset_le s).trans_lt (Cardinal.nat_lt_aleph0 _)


/-- The rank of a set of vectors as a natural number. -/
protected noncomputable def Set.finrank (s : Set M) : ℕ :=
  finrank R (span R s)


theorem finrank_span_le_card (s : Set M) [Fintype s] : finrank R (span R s) ≤ s.toFinset.card :=
                            /-
                              R : Type u
                              M : Type v
                              inst✝⁴ : Ring R
                              inst✝³ : AddCommGroup M
                              inst✝² : Module R M
                              inst✝¹ : StrongRankCondition R
                              s : Set M
                              inst✝ : Fintype ↑s
                              ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (Submodule.span R s) x …
                            -/
  finrank_le_of_rank_le (by simpa using rank_span_le (R := R) s)
                            /-
                              🎉 no goals
                            -/


theorem finrank_span_finset_le_card (s : Finset M) : (s : Set M).finrank R ≤ s.card :=
  calc
    (s : Set M).finrank R ≤ (s : Set M).toFinset.card := finrank_span_le_card (M := M) s
                     /-
                       R : Type u
                       M : Type v
                       inst✝³ : Ring R
                       inst✝² : AddCommGroup M
                       inst✝¹ : Module R M
                       inst✝ : StrongRankCondition R
                       s : Finset M
                       ⊢ Eq (↑s).toFinset.card s.card
                     -/
    _ = s.card := by simp
                     /-
                       🎉 no goals
                     -/


theorem finrank_range_le_card {ι : Type*} [Fintype ι] (b : ι → M) :
    (Set.range b).finrank R ≤ Fintype.card ι := by
  classical
  refine (finrank_span_le_card _).trans ?_
  rw [Set.toFinset_range]
  exact Finset.card_image_le


theorem finrank_span_eq_card [Nontrivial R] {ι : Type*} [Fintype ι] {b : ι → M}
    (hb : LinearIndependent R b) :
    finrank R (span R (Set.range b)) = Fintype.card ι :=
  finrank_eq_of_rank_eq
    (by
      /-
        R : Type u
        M : Type v
        inst✝⁵ : Ring R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : StrongRankCondition R
        inst✝¹ : Nontrivial R
        ι : Type u_2
        inst✝ : Fintype ι
        b : ι → M
        hb : LinearIndependent R b
        ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Submodule.span R (Set.ra …
      -/
      have : Module.rank R (span R (Set.range b)) = #(Set.range b) := rank_span hb
      rwa [← lift_inj, mk_range_eq_of_injective hb.injective, Cardinal.mk_fintype, lift_natCast,
        lift_eq_nat_iff] at this)


theorem finrank_span_set_eq_card {s : Set M} [Fintype s] (hs : LinearIndependent R ((↑) : s → M)) :
    finrank R (span R s) = s.toFinset.card :=
  finrank_eq_of_rank_eq
    (by
      /-
        R : Type u
        M : Type v
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : StrongRankCondition R
        s : Set M
        inst✝ : Fintype ↑s
        hs : LinearIndependent R Subtype.val
        ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Submodule.span R s) x))  …
      -/
      have : Module.rank R (span R s) = #s := rank_span_set hs
      /-
        R : Type u
        M : Type v
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : StrongRankCondition R
        s : Set M
        inst✝ : Fintype ↑s
        hs : LinearIndependent R Subtype.val
        this : Eq (Module.rank R (Subtype fun x => Membership.mem (Submodule.span R s) …
        ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Submodule.span R s) x))  …
      -/
      rwa [Cardinal.mk_fintype, ← Set.toFinset_card] at this)
      /-
        🎉 no goals
      -/


theorem finrank_span_finset_eq_card {s : Finset M} (hs : LinearIndependent R ((↑) : s → M)) :
    finrank R (span R (s : Set M)) = s.card := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    s : Finset M
    hs : LinearIndependent R Subtype.val
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (Submodule.span R ↑s)  …
  -/
  convert finrank_span_set_eq_card (s := (s : Set M)) hs
  /-
    case h.e'_3.h.e'_2
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    s : Finset M
    hs : LinearIndependent R Subtype.val
    ⊢ Eq s (↑s).toFinset
  -/
  ext
  /-
    case h.e'_3.h.e'_2.h
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : StrongRankCondition R
    s : Finset M
    hs : LinearIndependent R Subtype.val
    a✝ : M
    ⊢ Iff (Membership.mem s a✝) (Membership.mem (↑s).toFinset a✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem span_lt_of_subset_of_card_lt_finrank {s : Set M} [Fintype s] {t : Submodule R M}
    (subset : s ⊆ t) (card_lt : s.toFinset.card < finrank R t) : span R s < t :=
  lt_of_le_of_finrank_lt_finrank (span_le.mpr subset)
    (lt_of_le_of_lt (finrank_span_le_card _) card_lt)


theorem span_lt_top_of_card_lt_finrank {s : Set M} [Fintype s]
    (card_lt : s.toFinset.card < finrank R M) : span R s < ⊤ :=
  lt_top_of_finrank_lt_finrank (lt_of_le_of_lt (finrank_span_le_card _) card_lt)


lemma finrank_le_of_span_eq_top {ι : Type*} [Fintype ι] {v : ι → M}
    (hv : Submodule.span R (Set.range v) = ⊤) : finrank R M ≤ Fintype.card ι := by
  classical
  rw [← finrank_top, ← hv]
  exact (finrank_span_le_card _).trans (by convert Fintype.card_range_le v; rw [Set.toFinset_card])


@[simp]
theorem Subalgebra.rank_toSubmodule (S : Subalgebra F E) :
    Module.rank F (Subalgebra.toSubmodule S) = Module.rank F S :=
  rfl


@[simp]
theorem Subalgebra.finrank_toSubmodule (S : Subalgebra F E) :
    finrank F (Subalgebra.toSubmodule S) = finrank F S :=
  rfl


theorem subalgebra_top_rank_eq_submodule_top_rank :
    Module.rank F (⊤ : Subalgebra F E) = Module.rank F (⊤ : Submodule F E) := by
  /-
    F : Type u_2
    E : Type u_3
    inst✝² : CommRing F
    inst✝¹ : Ring E
    inst✝ : Algebra F E
    ⊢ Eq (Module.rank F (Subtype fun x => Membership.mem Top.top x)) (Module.rank  …
  -/
  rw [← Algebra.top_toSubmodule]
  /-
    F : Type u_2
    E : Type u_3
    inst✝² : CommRing F
    inst✝¹ : Ring E
    inst✝ : Algebra F E
    ⊢ Eq (Module.rank F (Subtype fun x => Membership.mem Top.top x)) (Module.rank  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem subalgebra_top_finrank_eq_submodule_top_finrank :
    finrank F (⊤ : Subalgebra F E) = finrank F (⊤ : Submodule F E) := by
  /-
    F : Type u_2
    E : Type u_3
    inst✝² : CommRing F
    inst✝¹ : Ring E
    inst✝ : Algebra F E
    ⊢ Eq (Module.finrank F (Subtype fun x => Membership.mem Top.top x)) (Module.fi …
  -/
  rw [← Algebra.top_toSubmodule]
  /-
    F : Type u_2
    E : Type u_3
    inst✝² : CommRing F
    inst✝¹ : Ring E
    inst✝ : Algebra F E
    ⊢ Eq (Module.finrank F (Subtype fun x => Membership.mem Top.top x)) (Module.fi …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Subalgebra.rank_top : Module.rank F (⊤ : Subalgebra F E) = Module.rank F E := by
  /-
    F : Type u_2
    E : Type u_3
    inst✝² : CommRing F
    inst✝¹ : Ring E
    inst✝ : Algebra F E
    ⊢ Eq (Module.rank F (Subtype fun x => Membership.mem Top.top x)) (Module.rank  …
  -/
  rw [subalgebra_top_rank_eq_submodule_top_rank]
  /-
    F : Type u_2
    E : Type u_3
    inst✝² : CommRing F
    inst✝¹ : Ring E
    inst✝ : Algebra F E
    ⊢ Eq (Module.rank F (Subtype fun x => Membership.mem Top.top x)) (Module.rank  …
  -/
  exact _root_.rank_top F E
  /-
    🎉 no goals
  -/


@[simp]
theorem Subalgebra.rank_bot : Module.rank F (⊥ : Subalgebra F E) = 1 :=
  (Subalgebra.toSubmoduleEquiv (⊥ : Subalgebra F E)).symm.rank_eq.trans <| by
    /-
      F : Type u_2
      E : Type u_3
      inst✝⁵ : CommRing F
      inst✝⁴ : Ring E
      inst✝³ : Algebra F E
      inst✝² : StrongRankCondition F
      inst✝¹ : NoZeroSMulDivisors F E
      inst✝ : Nontrivial E
      ⊢ Eq (Module.rank F (Subtype fun x => Membership.mem (Subalgebra.toSubmodule B …
    -/
    rw [Algebra.toSubmodule_bot, one_eq_span, rank_span_set, mk_singleton _]
    /-
      F : Type u_2
      E : Type u_3
      inst✝⁵ : CommRing F
      inst✝⁴ : Ring E
      inst✝³ : Algebra F E
      inst✝² : StrongRankCondition F
      inst✝¹ : NoZeroSMulDivisors F E
      inst✝ : Nontrivial E
      ⊢ LinearIndependent F fun x => ↑x
    -/
    letI := Module.nontrivial F E
    /-
      F : Type u_2
      E : Type u_3
      inst✝⁵ : CommRing F
      inst✝⁴ : Ring E
      inst✝³ : Algebra F E
      inst✝² : StrongRankCondition F
      inst✝¹ : NoZeroSMulDivisors F E
      inst✝ : Nontrivial E
      this : Nontrivial F := Module.nontrivial F E
      ⊢ LinearIndependent F fun x => ↑x
    -/
    exact linearIndependent_singleton one_ne_zero
    /-
      🎉 no goals
    -/


@[simp]
theorem Subalgebra.finrank_bot : finrank F (⊥ : Subalgebra F E) = 1 :=
                            /-
                              F : Type u_2
                              E : Type u_3
                              inst✝⁵ : CommRing F
                              inst✝⁴ : Ring E
                              inst✝³ : Algebra F E
                              inst✝² : StrongRankCondition F
                              inst✝¹ : NoZeroSMulDivisors F E
                              inst✝ : Nontrivial E
                              ⊢ Eq (Module.rank F (Subtype fun x => Membership.mem Bot.bot x)) ↑1
                            -/
  finrank_eq_of_rank_eq (by simp)
                            /-
                              🎉 no goals
                            -/


