theorem rank_quotient_eq_of_le_torsion {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    {M' : Submodule R M} (hN : M' ≤ torsion R M) : Module.rank R (M ⧸ M') = Module.rank R M :=
  (rank_quotient_le M').antisymm <| by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      M' : Submodule R M
      hN : LE.le M' (Submodule.torsion R M)
      ⊢ LE.le (Module.rank R M) (Module.rank R (HasQuotient.Quotient M M'))
    -/
    nontriviality R
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      M' : Submodule R M
      hN : LE.le M' (Submodule.torsion R M)
      a✝ : Nontrivial R
      ⊢ LE.le (Module.rank R M) (Module.rank R (HasQuotient.Quotient M M'))
    -/
    rw [Module.rank]
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      M' : Submodule R M
      hN : LE.le M' (Submodule.torsion R M)
      a✝ : Nontrivial R
      ⊢ LE.le (iSup fun ι => Cardinal.mk ↑↑ι) (Module.rank R (HasQuotient.Quotient M …
    -/
    have := nonempty_linearIndependent_set R M
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      M' : Submodule R M
      hN : LE.le M' (Submodule.torsion R M)
      a✝ : Nontrivial R
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      ⊢ LE.le (iSup fun ι => Cardinal.mk ↑↑ι) (Module.rank R (HasQuotient.Quotient M …
    -/
    refine ciSup_le fun ⟨s, hs⟩ ↦ LinearIndependent.cardinal_le_rank (v := (M'.mkQ ·)) ?_
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      M' : Submodule R M
      hN : LE.le M' (Submodule.torsion R M)
      a✝ : Nontrivial R
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      x✝ : Subtype fun s => LinearIndependent R Subtype.val
      s : Set M
      hs : LinearIndependent R Subtype.val
      ⊢ LinearIndependent R fun x => M'.mkQ ↑x
    -/
    rw [linearIndependent_iff'] at hs ⊢
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      M' : Submodule R M
      hN : LE.le M' (Submodule.torsion R M)
      a✝ : Nontrivial R
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      x✝ : Subtype fun s => LinearIndependent R Subtype.val
      s : Set M
      hs✝ : LinearIndependent R Subtype.val
      hs : ∀ (s_1 : Finset (Subtype fun x => Membership.mem s x)) (g : (Subtype fun  …
      ⊢ ∀ (s_1 : Finset ↑↑⟨s, hs✝⟩) (g : ↑↑⟨s, hs✝⟩ → R), Eq (s_1.sum fun i => HSMul …
    -/
    simp_rw [← map_smul, ← map_sum, mkQ_apply, Quotient.mk_eq_zero]
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      M' : Submodule R M
      hN : LE.le M' (Submodule.torsion R M)
      a✝ : Nontrivial R
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      x✝ : Subtype fun s => LinearIndependent R Subtype.val
      s : Set M
      hs✝ : LinearIndependent R Subtype.val
      hs : ∀ (s_1 : Finset (Subtype fun x => Membership.mem s x)) (g : (Subtype fun  …
      ⊢ ∀ (s_1 : Finset ↑s) (g : ↑s → R), Membership.mem M' (s_1.sum fun x => HSMul. …
    -/
    intro t g hg i hi
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      M' : Submodule R M
      hN : LE.le M' (Submodule.torsion R M)
      a✝ : Nontrivial R
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      x✝ : Subtype fun s => LinearIndependent R Subtype.val
      s : Set M
      hs✝ : LinearIndependent R Subtype.val
      hs : ∀ (s_1 : Finset (Subtype fun x => Membership.mem s x)) (g : (Subtype fun  …
      t : Finset ↑s
      g : ↑s → R
      hg : Membership.mem M' (t.sum fun x => HSMul.hSMul (g x) ↑x)
      i : ↑s
      hi : Membership.mem t i
      ⊢ Eq (g i) 0
    -/
    obtain ⟨r, hg⟩ := hN hg
    /-
      case intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      M' : Submodule R M
      hN : LE.le M' (Submodule.torsion R M)
      a✝ : Nontrivial R
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      x✝ : Subtype fun s => LinearIndependent R Subtype.val
      s : Set M
      hs✝ : LinearIndependent R Subtype.val
      hs : ∀ (s_1 : Finset (Subtype fun x => Membership.mem s x)) (g : (Subtype fun  …
      t : Finset ↑s
      g : ↑s → R
      hg✝ : Membership.mem M' (t.sum fun x => HSMul.hSMul (g x) ↑x)
      i : ↑s
      hi : Membership.mem t i
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      hg : Eq (HSMul.hSMul r (t.sum fun x => HSMul.hSMul (g x) ↑x)) 0
      ⊢ Eq (g i) 0
    -/
    simp_rw [Finset.smul_sum, Submonoid.smul_def, smul_smul] at hg
    /-
      case intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      M' : Submodule R M
      hN : LE.le M' (Submodule.torsion R M)
      a✝ : Nontrivial R
      this : Nonempty (Subtype fun s => LinearIndependent R Subtype.val)
      x✝ : Subtype fun s => LinearIndependent R Subtype.val
      s : Set M
      hs✝ : LinearIndependent R Subtype.val
      hs : ∀ (s_1 : Finset (Subtype fun x => Membership.mem s x)) (g : (Subtype fun  …
      t : Finset ↑s
      g : ↑s → R
      hg✝ : Membership.mem M' (t.sum fun x => HSMul.hSMul (g x) ↑x)
      i : ↑s
      hi : Membership.mem t i
      r : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      hg : Eq (t.sum fun x => HSMul.hSMul (HMul.hMul (↑r) (g x)) ↑x) 0
      ⊢ Eq (g i) 0
    -/
    exact r.prop _ (mul_comm (g i) r ▸ hs t _ hg i hi)
    /-
      🎉 no goals
    -/

