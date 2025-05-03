/-- If the support of each element in a generating set of a permutation group is finite,
then the support of every element in the group is finite. -/
theorem finite_compl_fixedBy_closure_iff {S : Set G} :
    (∀ g ∈ closure S, (fixedBy α g)ᶜ.Finite) ↔ ∀ g ∈ S, (fixedBy α g)ᶜ.Finite :=
  ⟨fun h g hg ↦ h g (subset_closure hg), fun h g hg ↦ by
    refine closure_induction h (by simp) (fun g g' _ _ hg hg' ↦ (hg.union hg').subset ?_)
      (by simp) hg
    /-
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      S : Set G
      h : ∀ (g : G), Membership.mem S g → (HasCompl.compl (MulAction.fixedBy α g)).F …
      g✝ : G
      hg✝ : Membership.mem (Subgroup.closure S) g✝
      g g' : G
      x✝¹ : Membership.mem (Subgroup.closure S) g
      x✝ : Membership.mem (Subgroup.closure S) g'
      hg : (HasCompl.compl (MulAction.fixedBy α g)).Finite
      hg' : (HasCompl.compl (MulAction.fixedBy α g')).Finite
      ⊢ HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α (HMul.hMul g g'))) (Un …
    -/
    simp_rw [← compl_inter, compl_subset_compl, fixedBy_mul]⟩
    /-
      🎉 no goals
    -/


/-- Given a symmetric generating set of a permutation group, if T is a nonempty proper subset of
an orbit, then there exists a generator that sends some element of T into the complement of T. -/
theorem exists_smul_not_mem_of_subset_orbit_closure (S : Set G) (T : Set α) {a : α}
    (hS : ∀ g ∈ S, g⁻¹ ∈ S) (subset : T ⊆ orbit (closure S) a) (not_mem : a ∉ T)
    (nonempty : T.Nonempty) : ∃ σ ∈ S, ∃ a ∈ T, σ • a ∉ T := by
  have key0 : ¬ closure S ≤ stabilizer G T := by
    have ⟨b, hb⟩ := nonempty
    obtain ⟨σ, rfl⟩ := subset hb
    contrapose! not_mem with h
    exact smul_mem_smul_set_iff.mp ((h σ.2).symm ▸ hb)
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    S : Set G
    T : Set α
    a : α
    hS : ∀ (g : G), Membership.mem S g → Membership.mem S (Inv.inv g)
    subset : HasSubset.Subset T (MulAction.orbit (Subtype fun x => Membership.mem  …
    not_mem : Not (Membership.mem T a)
    nonempty : T.Nonempty
    key0 : Not (LE.le (Subgroup.closure S) (MulAction.stabilizer G T))
    ⊢ Exists fun σ => And (Membership.mem S σ) (Exists fun a => And (Membership.me …
  -/
  contrapose! key0
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    S : Set G
    T : Set α
    a : α
    hS : ∀ (g : G), Membership.mem S g → Membership.mem S (Inv.inv g)
    subset : HasSubset.Subset T (MulAction.orbit (Subtype fun x => Membership.mem  …
    not_mem : Not (Membership.mem T a)
    nonempty : T.Nonempty
    key0 : ∀ (σ : G), Membership.mem S σ → ∀ (a : α), Membership.mem T a → Members …
    ⊢ LE.le (Subgroup.closure S) (MulAction.stabilizer G T)
  -/
  refine (closure_le _).mpr fun σ hσ ↦ ?_
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    S : Set G
    T : Set α
    a : α
    hS : ∀ (g : G), Membership.mem S g → Membership.mem S (Inv.inv g)
    subset : HasSubset.Subset T (MulAction.orbit (Subtype fun x => Membership.mem  …
    not_mem : Not (Membership.mem T a)
    nonempty : T.Nonempty
    key0 : ∀ (σ : G), Membership.mem S σ → ∀ (a : α), Membership.mem T a → Members …
    σ : G
    hσ : Membership.mem S σ
    ⊢ Membership.mem (↑(MulAction.stabilizer G T)) σ
  -/
  simp_rw [SetLike.mem_coe, mem_stabilizer_iff, Set.ext_iff, mem_smul_set_iff_inv_smul_mem]
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    S : Set G
    T : Set α
    a : α
    hS : ∀ (g : G), Membership.mem S g → Membership.mem S (Inv.inv g)
    subset : HasSubset.Subset T (MulAction.orbit (Subtype fun x => Membership.mem  …
    not_mem : Not (Membership.mem T a)
    nonempty : T.Nonempty
    key0 : ∀ (σ : G), Membership.mem S σ → ∀ (a : α), Membership.mem T a → Members …
    σ : G
    hσ : Membership.mem S σ
    ⊢ ∀ (x : α), Iff (Membership.mem T (HSMul.hSMul (Inv.inv σ) x)) (Membership.me …
  -/
  exact fun a ↦ ⟨fun h ↦ smul_inv_smul σ a ▸ key0 σ hσ (σ⁻¹ • a) h, key0 σ⁻¹ (hS σ hσ) a⟩
  /-
    🎉 no goals
  -/


theorem finite_compl_fixedBy_swap {x y : α} : (fixedBy α (swap x y))ᶜ.Finite :=
                                      /-
                                        α : Type u_2
                                        inst✝ : DecidableEq α
                                        x y : α
                                        ⊢ (Insert.insert x (Singleton.singleton y)).Finite
                                      -/
  Set.Finite.subset (s := {x, y}) (by simp)
                                      /-
                                        🎉 no goals
                                      -/
                                       /-
                                         α : Type u_2
                                         inst✝ : DecidableEq α
                                         x y z : α
                                         h : Membership.mem (HasCompl.compl (Insert.insert x (Singleton.singleton y))) z
                                         ⊢ Membership.mem (MulAction.fixedBy α (Equiv.swap x y)) z
                                       -/
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
    (compl_subset_comm.mp fun z h ↦ by apply swap_apply_of_ne_of_ne <;> rintro rfl <;> simp at h)
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


theorem Equiv.Perm.IsSwap.finite_compl_fixedBy {σ : Perm α} (h : σ.IsSwap) :
    (fixedBy α σ)ᶜ.Finite := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h : σ.IsSwap
    ⊢ (HasCompl.compl (MulAction.fixedBy α σ)).Finite
  -/
  obtain ⟨x, y, -, rfl⟩ := h
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝ : DecidableEq α
    x y : α
    ⊢ (HasCompl.compl (MulAction.fixedBy α (Equiv.swap x y))).Finite
  -/
  exact finite_compl_fixedBy_swap
  /-
    🎉 no goals
  -/

-- this result cannot be moved to Perm/Basic since Perm/Basic is not allowed to import Submonoid

theorem SubmonoidClass.swap_mem_trans {a b c : α} {C} [SetLike C (Perm α)]
    [SubmonoidClass C (Perm α)] (M : C) (hab : swap a b ∈ M) (hbc : swap b c ∈ M) :
    swap a c ∈ M := by
  /-
    α : Type u_2
    inst✝² : DecidableEq α
    a b c : α
    C : Type u_3
    inst✝¹ : SetLike C (Equiv.Perm α)
    inst✝ : SubmonoidClass C (Equiv.Perm α)
    M : C
    hab : Membership.mem M (Equiv.swap a b)
    hbc : Membership.mem M (Equiv.swap b c)
    ⊢ Membership.mem M (Equiv.swap a c)
  -/
  obtain rfl | hab' := eq_or_ne a b
    /-
      case inl
      α : Type u_2
      inst✝² : DecidableEq α
      a c : α
      C : Type u_3
      inst✝¹ : SetLike C (Equiv.Perm α)
      inst✝ : SubmonoidClass C (Equiv.Perm α)
      M : C
      hab : Membership.mem M (Equiv.swap a a)
      hbc : Membership.mem M (Equiv.swap a c)
      ⊢ Membership.mem M (Equiv.swap a c)
    -/
  · exact hbc
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    inst✝² : DecidableEq α
    a b c : α
    C : Type u_3
    inst✝¹ : SetLike C (Equiv.Perm α)
    inst✝ : SubmonoidClass C (Equiv.Perm α)
    M : C
    hab : Membership.mem M (Equiv.swap a b)
    hbc : Membership.mem M (Equiv.swap b c)
    hab' : Ne a b
    ⊢ Membership.mem M (Equiv.swap a c)
  -/
  obtain rfl | hac := eq_or_ne a c
    /-
      case inr.inl
      α : Type u_2
      inst✝² : DecidableEq α
      a b : α
      C : Type u_3
      inst✝¹ : SetLike C (Equiv.Perm α)
      inst✝ : SubmonoidClass C (Equiv.Perm α)
      M : C
      hab : Membership.mem M (Equiv.swap a b)
      hab' : Ne a b
      hbc : Membership.mem M (Equiv.swap b a)
      ⊢ Membership.mem M (Equiv.swap a a)
    -/
  · exact swap_self a ▸ one_mem M
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u_2
    inst✝² : DecidableEq α
    a b c : α
    C : Type u_3
    inst✝¹ : SetLike C (Equiv.Perm α)
    inst✝ : SubmonoidClass C (Equiv.Perm α)
    M : C
    hab : Membership.mem M (Equiv.swap a b)
    hbc : Membership.mem M (Equiv.swap b c)
    hab' : Ne a b
    hac : Ne a c
    ⊢ Membership.mem M (Equiv.swap a c)
  -/
  rw [swap_comm, ← swap_mul_swap_mul_swap hab' hac]
  /-
    case inr.inr
    α : Type u_2
    inst✝² : DecidableEq α
    a b c : α
    C : Type u_3
    inst✝¹ : SetLike C (Equiv.Perm α)
    inst✝ : SubmonoidClass C (Equiv.Perm α)
    M : C
    hab : Membership.mem M (Equiv.swap a b)
    hbc : Membership.mem M (Equiv.swap b c)
    hab' : Ne a b
    hac : Ne a c
    ⊢ Membership.mem M (HMul.hMul (HMul.hMul (Equiv.swap b c) (Equiv.swap a b)) (E …
  -/
  exact mul_mem (mul_mem hbc hab) hbc
  /-
    🎉 no goals
  -/


/-- If a subgroup is generated by transpositions, then a transposition `swap x y` lies in the
  subgroup if and only if `x` lies in the same orbit as `y`. -/
theorem swap_mem_closure_isSwap {S : Set (Perm α)} (hS : ∀ f ∈ S, f.IsSwap) {x y : α} :
    swap x y ∈ closure S ↔ x ∈ orbit (closure S) y := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    S : Set (Equiv.Perm α)
    hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
    x y : α
    ⊢ Iff (Membership.mem (Subgroup.closure S) (Equiv.swap x y)) (Membership.mem ( …
  -/
  refine ⟨fun h ↦ ⟨⟨swap x y, h⟩, swap_apply_right x y⟩, fun hf ↦ ?_⟩
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    S : Set (Equiv.Perm α)
    hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
    x y : α
    hf : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgrou …
    ⊢ Membership.mem (Subgroup.closure S) (Equiv.swap x y)
  -/
  by_contra h
  have := exists_smul_not_mem_of_subset_orbit_closure S {x | swap x y ∈ closure S}
    (fun f hf ↦ ?_) (fun z hz ↦ ?_) h ⟨y, ?_⟩
    /-
      case refine_4
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      x y : α
      hf : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgrou …
      h : Not (Membership.mem (Subgroup.closure S) (Equiv.swap x y))
      this : Exists fun σ => And (Membership.mem S σ) (Exists fun a => And (Membersh …
      ⊢ False
    -/
  · obtain ⟨σ, hσ, a, ha, hσa⟩ := this
    /-
      case refine_4.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      x y : α
      hf : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgrou …
      h : Not (Membership.mem (Subgroup.closure S) (Equiv.swap x y))
      σ : Equiv.Perm α
      hσ : Membership.mem S σ
      a : α
      ha : Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S) (Equiv …
      hσa : Not (Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S)  …
      ⊢ False
    -/
    obtain ⟨z, w, hzw, rfl⟩ := hS σ hσ
    /-
      case refine_4.intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      x y : α
      hf : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgrou …
      h : Not (Membership.mem (Subgroup.closure S) (Equiv.swap x y))
      a : α
      ha : Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S) (Equiv …
      z w : α
      hzw : Ne z w
      hσ : Membership.mem S (Equiv.swap z w)
      hσa : Not (Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S)  …
      ⊢ False
    -/
    have := ne_of_mem_of_not_mem ha hσa
    /-
      case refine_4.intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      x y : α
      hf : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgrou …
      h : Not (Membership.mem (Subgroup.closure S) (Equiv.swap x y))
      a : α
      ha : Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S) (Equiv …
      z w : α
      hzw : Ne z w
      hσ : Membership.mem S (Equiv.swap z w)
      hσa : Not (Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S)  …
      this : Ne a (HSMul.hSMul (Equiv.swap z w) a)
      ⊢ False
    -/
    rw [Perm.smul_def, ne_comm, swap_apply_ne_self_iff, and_iff_right hzw] at this
    /-
      case refine_4.intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      x y : α
      hf : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgrou …
      h : Not (Membership.mem (Subgroup.closure S) (Equiv.swap x y))
      a : α
      ha : Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S) (Equiv …
      z w : α
      hzw : Ne z w
      hσ : Membership.mem S (Equiv.swap z w)
      hσa : Not (Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S)  …
      this : Or (Eq a z) (Eq a w)
      ⊢ False
    -/
    refine hσa (SubmonoidClass.swap_mem_trans (closure S) ?_ ha)
    /-
      case refine_4.intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      x y : α
      hf : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgrou …
      h : Not (Membership.mem (Subgroup.closure S) (Equiv.swap x y))
      a : α
      ha : Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S) (Equiv …
      z w : α
      hzw : Ne z w
      hσ : Membership.mem S (Equiv.swap z w)
      hσa : Not (Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S)  …
      this : Or (Eq a z) (Eq a w)
      ⊢ Membership.mem (Subgroup.closure S) (Equiv.swap (HSMul.hSMul (Equiv.swap z w …
    -/
                                 /-
                                   🎉 no goals
                                 -/
    obtain rfl | rfl := this <;> simpa [swap_comm] using subset_closure hσ
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case refine_1
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      x y : α
      hf✝ : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgro …
      h : Not (Membership.mem (Subgroup.closure S) (Equiv.swap x y))
      f : Equiv.Perm α
      hf : Membership.mem S f
      ⊢ Membership.mem S (Inv.inv f)
    -/
  · obtain ⟨x, y, -, rfl⟩ := hS f hf; rwa [swap_inv]
                                      /-
                                        🎉 no goals
                                      -/
    /-
      case refine_2
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      x y : α
      hf : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgrou …
      h : Not (Membership.mem (Subgroup.closure S) (Equiv.swap x y))
      z : α
      hz : Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S) (Equiv …
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgroup.c …
    -/
  · exact orbit_eq_iff.mpr hf ▸ ⟨⟨swap z y, hz⟩, swap_apply_right z y⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      x y : α
      hf : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgrou …
      h : Not (Membership.mem (Subgroup.closure S) (Equiv.swap x y))
      ⊢ Membership.mem (setOf fun x => Membership.mem (Subgroup.closure S) (Equiv.sw …
    -/
  · rw [mem_setOf, swap_self]; apply one_mem
                               /-
                                 🎉 no goals
                               -/


/-- If a subgroup is generated by transpositions, then a permutation `f` lies in the subgroup if
  and only if `f` has finite support and `f x` always lies in the same orbit as `x`. -/
theorem mem_closure_isSwap {S : Set (Perm α)} (hS : ∀ f ∈ S, f.IsSwap) {f : Perm α} :
    f ∈ closure S ↔ (fixedBy α f)ᶜ.Finite ∧ ∀ x, f x ∈ orbit (closure S) x := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    S : Set (Equiv.Perm α)
    hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
    f : Equiv.Perm α
    ⊢ Iff (Membership.mem (Subgroup.closure S) f) (And (HasCompl.compl (MulAction. …
  -/
  refine ⟨fun hf ↦ ⟨?_, fun x ↦ mem_orbit_iff.mpr ⟨⟨f, hf⟩, rfl⟩⟩, ?_⟩
    /-
      case refine_1
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      f : Equiv.Perm α
      hf : Membership.mem (Subgroup.closure S) f
      ⊢ (HasCompl.compl (MulAction.fixedBy α f)).Finite
    -/
  · exact finite_compl_fixedBy_closure_iff.mpr (fun f hf ↦ (hS f hf).finite_compl_fixedBy) _ hf
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_2
    inst✝ : DecidableEq α
    S : Set (Equiv.Perm α)
    hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
    f : Equiv.Perm α
    ⊢ And (HasCompl.compl (MulAction.fixedBy α f)).Finite (∀ (x : α), Membership.m …
  -/
  rintro ⟨fin, hf⟩
  /-
    case refine_2.intro
    α : Type u_2
    inst✝ : DecidableEq α
    S : Set (Equiv.Perm α)
    hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
    f : Equiv.Perm α
    fin : (HasCompl.compl (MulAction.fixedBy α f)).Finite
    hf : ∀ (x : α), Membership.mem (MulAction.orbit (Subtype fun x => Membership.m …
    ⊢ Membership.mem (Subgroup.closure S) f
  -/
  set supp := (fixedBy α f)ᶜ with supp_eq
  /-
    case refine_2.intro
    α : Type u_2
    inst✝ : DecidableEq α
    S : Set (Equiv.Perm α)
    hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
    f : Equiv.Perm α
    hf : ∀ (x : α), Membership.mem (MulAction.orbit (Subtype fun x => Membership.m …
    supp : Set α := HasCompl.compl (MulAction.fixedBy α f)
    fin : supp.Finite
    supp_eq : Eq supp (HasCompl.compl (MulAction.fixedBy α f))
    ⊢ Membership.mem (Subgroup.closure S) f
  -/
  suffices h : (fixedBy α f)ᶜ ⊆ supp → f ∈ closure S from h supp_eq.symm.subset
  /-
    case refine_2.intro
    α : Type u_2
    inst✝ : DecidableEq α
    S : Set (Equiv.Perm α)
    hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
    f : Equiv.Perm α
    hf : ∀ (x : α), Membership.mem (MulAction.orbit (Subtype fun x => Membership.m …
    supp : Set α := HasCompl.compl (MulAction.fixedBy α f)
    fin : supp.Finite
    supp_eq : Eq supp (HasCompl.compl (MulAction.fixedBy α f))
    ⊢ HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α f)) supp → Membership. …
  -/
  clear_value supp; clear supp_eq; revert f
  /-
    case refine_2.intro
    α : Type u_2
    inst✝ : DecidableEq α
    S : Set (Equiv.Perm α)
    hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
    supp : Set α
    fin : supp.Finite
    ⊢ ∀ {f : Equiv.Perm α}, (∀ (x : α), Membership.mem (MulAction.orbit (Subtype f …
  -/
  apply fin.induction_on ..
    /-
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      supp : Set α
      fin : supp.Finite
      ⊢ ∀ {f : Equiv.Perm α}, (∀ (x : α), Membership.mem (MulAction.orbit (Subtype f …
    -/
  · rintro f - emp; convert (closure S).one_mem; ext; by_contra h; exact emp h
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    S : Set (Equiv.Perm α)
    hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
    supp : Set α
    fin : supp.Finite
    ⊢ ∀ {a : α} {s : Set α}, Not (Membership.mem s a) → s.Finite → (∀ {f : Equiv.P …
  -/
  rintro a s - - ih f hf supp_subset
  refine (mul_mem_cancel_left ((swap_mem_closure_isSwap hS).2 (hf a))).1
    (ih (fun b ↦ ?_) fun b hb ↦ ?_)
    /-
      case refine_1
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      supp : Set α
      fin : supp.Finite
      a : α
      s : Set α
      ih : ∀ {f : Equiv.Perm α}, (∀ (x : α), Membership.mem (MulAction.orbit (Subtyp …
      f : Equiv.Perm α
      hf : ∀ (x : α), Membership.mem (MulAction.orbit (Subtype fun x => Membership.m …
      supp_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α f)) (Inser …
      b : α
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgroup.c …
    -/
  · rw [Perm.mul_apply, swap_apply_def]; split_ifs with h1 h2
      /-
        case pos
        α : Type u_2
        inst✝ : DecidableEq α
        S : Set (Equiv.Perm α)
        hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
        supp : Set α
        fin : supp.Finite
        a : α
        s : Set α
        ih : ∀ {f : Equiv.Perm α}, (∀ (x : α), Membership.mem (MulAction.orbit (Subtyp …
        f : Equiv.Perm α
        hf : ∀ (x : α), Membership.mem (MulAction.orbit (Subtype fun x => Membership.m …
        supp_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α f)) (Inser …
        b : α
        h1 : Eq (f b) (f a)
        ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgroup.c …
      -/
    · rw [← orbit_eq_iff.mpr (hf b), h1, orbit_eq_iff.mpr (hf a)]; apply mem_orbit_self
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
      /-
        case pos
        α : Type u_2
        inst✝ : DecidableEq α
        S : Set (Equiv.Perm α)
        hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
        supp : Set α
        fin : supp.Finite
        a : α
        s : Set α
        ih : ∀ {f : Equiv.Perm α}, (∀ (x : α), Membership.mem (MulAction.orbit (Subtyp …
        f : Equiv.Perm α
        hf : ∀ (x : α), Membership.mem (MulAction.orbit (Subtype fun x => Membership.m …
        supp_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α f)) (Inser …
        b : α
        h1 : Not (Eq (f b) (f a))
        h2 : Eq (f b) a
        ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgroup.c …
      -/
    · rw [← orbit_eq_iff.mpr (hf b), h2]; apply hf
                                          /-
                                            🎉 no goals
                                          -/
      /-
        case neg
        α : Type u_2
        inst✝ : DecidableEq α
        S : Set (Equiv.Perm α)
        hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
        supp : Set α
        fin : supp.Finite
        a : α
        s : Set α
        ih : ∀ {f : Equiv.Perm α}, (∀ (x : α), Membership.mem (MulAction.orbit (Subtyp …
        f : Equiv.Perm α
        hf : ∀ (x : α), Membership.mem (MulAction.orbit (Subtype fun x => Membership.m …
        supp_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α f)) (Inser …
        b : α
        h1 : Not (Eq (f b) (f a))
        h2 : Not (Eq (f b) a)
        ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgroup.c …
      -/
    · exact hf b
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      supp : Set α
      fin : supp.Finite
      a : α
      s : Set α
      ih : ∀ {f : Equiv.Perm α}, (∀ (x : α), Membership.mem (MulAction.orbit (Subtyp …
      f : Equiv.Perm α
      hf : ∀ (x : α), Membership.mem (MulAction.orbit (Subtype fun x => Membership.m …
      supp_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α f)) (Inser …
      b : α
      hb : Membership.mem (HasCompl.compl (MulAction.fixedBy α (HMul.hMul (Equiv.swa …
      ⊢ Membership.mem s b
    -/
  · contrapose! hb
    simp_rw [not_mem_compl_iff, mem_fixedBy, Perm.smul_def, Perm.mul_apply, swap_apply_def,
      apply_eq_iff_eq]
    /-
      case refine_2
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      supp : Set α
      fin : supp.Finite
      a : α
      s : Set α
      ih : ∀ {f : Equiv.Perm α}, (∀ (x : α), Membership.mem (MulAction.orbit (Subtyp …
      f : Equiv.Perm α
      hf : ∀ (x : α), Membership.mem (MulAction.orbit (Subtype fun x => Membership.m …
      supp_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α f)) (Inser …
      b : α
      hb : Not (Membership.mem s b)
      ⊢ Eq (ite (Eq b a) a (ite (Eq (f b) a) (f a) (f b))) b
    -/
    by_cases hb' : f b = b
      /-
        case pos
        α : Type u_2
        inst✝ : DecidableEq α
        S : Set (Equiv.Perm α)
        hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
        supp : Set α
        fin : supp.Finite
        a : α
        s : Set α
        ih : ∀ {f : Equiv.Perm α}, (∀ (x : α), Membership.mem (MulAction.orbit (Subtyp …
        f : Equiv.Perm α
        hf : ∀ (x : α), Membership.mem (MulAction.orbit (Subtype fun x => Membership.m …
        supp_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α f)) (Inser …
        b : α
        hb : Not (Membership.mem s b)
        hb' : Eq (f b) b
        ⊢ Eq (ite (Eq b a) a (ite (Eq (f b) a) (f a) (f b))) b
      -/
                                     /-
                                       🎉 no goals
                                     -/
    · rw [hb']; split_ifs with h <;> simp only [h]
                                     /-
                                       🎉 no goals
                                     -/
    /-
      case neg
      α : Type u_2
      inst✝ : DecidableEq α
      S : Set (Equiv.Perm α)
      hS : ∀ (f : Equiv.Perm α), Membership.mem S f → f.IsSwap
      supp : Set α
      fin : supp.Finite
      a : α
      s : Set α
      ih : ∀ {f : Equiv.Perm α}, (∀ (x : α), Membership.mem (MulAction.orbit (Subtyp …
      f : Equiv.Perm α
      hf : ∀ (x : α), Membership.mem (MulAction.orbit (Subtype fun x => Membership.m …
      supp_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α f)) (Inser …
      b : α
      hb : Not (Membership.mem s b)
      hb' : Not (Eq (f b) b)
      ⊢ Eq (ite (Eq b a) a (ite (Eq (f b) a) (f a) (f b))) b
    -/
    simp [show b = a by simpa [hb] using supp_subset hb']
    /-
      🎉 no goals
    -/


/-- A permutation is a product of transpositions if and only if it has finite support. -/
theorem mem_closure_isSwap' {f : Perm α} :
    f ∈ closure {σ : Perm α | σ.IsSwap} ↔ (fixedBy α f)ᶜ.Finite := by
  refine (mem_closure_isSwap fun _ ↦ id).trans
    (and_iff_left fun x ↦ ⟨⟨swap x (f x), ?_⟩, swap_apply_left x (f x)⟩)
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    x : α
    ⊢ Membership.mem (Subgroup.closure fun x => Exists fun x_1 => Exists fun y =>  …
  -/
  by_cases h : x = f x
    /-
      case pos
      α : Type u_2
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      h : Eq x (f x)
      ⊢ Membership.mem (Subgroup.closure fun x => Exists fun x_1 => Exists fun y =>  …
    -/
  · rw [← h, swap_self]
    /-
      case pos
      α : Type u_2
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      h : Eq x (f x)
      ⊢ Membership.mem (Subgroup.closure fun x => Exists fun x_1 => Exists fun y =>  …
    -/
    apply Subgroup.one_mem
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      h : Not (Eq x (f x))
      ⊢ Membership.mem (Subgroup.closure fun x => Exists fun x_1 => Exists fun y =>  …
    -/
  · exact subset_closure ⟨x, f x, h, rfl⟩
    /-
      🎉 no goals
    -/


/-- A transitive permutation group generated by transpositions must be the whole symmetric group -/
theorem closure_of_isSwap_of_isPretransitive [Finite α] {S : Set (Perm α)} (hS : ∀ σ ∈ S, σ.IsSwap)
    [MulAction.IsPretransitive (Subgroup.closure S) α] : Subgroup.closure S = ⊤ := by
  /-
    α : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : Finite α
    S : Set (Equiv.Perm α)
    hS : ∀ (σ : Equiv.Perm α), Membership.mem S σ → σ.IsSwap
    inst✝ : MulAction.IsPretransitive (Subtype fun x => Membership.mem (Subgroup.c …
    ⊢ Eq (Subgroup.closure S) Top.top
  -/
  simp [eq_top_iff', mem_closure_isSwap hS, orbit_eq_univ, Set.toFinite]
  /-
    🎉 no goals
  -/


/-- A transitive permutation group generated by transpositions must be the whole symmetric group -/
theorem surjective_of_isSwap_of_isPretransitive [Finite α] (S : Set G)
    (hS1 : ∀ σ ∈ S, Perm.IsSwap (MulAction.toPermHom G α σ)) (hS2 : Subgroup.closure S = ⊤)
    [h : MulAction.IsPretransitive G α] : Function.Surjective (MulAction.toPermHom G α) := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝³ : Group G
    inst✝² : MulAction G α
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    S : Set G
    hS1 : ∀ (σ : G), Membership.mem S σ → ((MulAction.toPermHom G α) σ).IsSwap
    hS2 : Eq (Subgroup.closure S) Top.top
    h : MulAction.IsPretransitive G α
    ⊢ Function.Surjective ⇑(MulAction.toPermHom G α)
  -/
  rw [← MonoidHom.range_eq_top]
  /-
    G : Type u_1
    α : Type u_2
    inst✝³ : Group G
    inst✝² : MulAction G α
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    S : Set G
    hS1 : ∀ (σ : G), Membership.mem S σ → ((MulAction.toPermHom G α) σ).IsSwap
    hS2 : Eq (Subgroup.closure S) Top.top
    h : MulAction.IsPretransitive G α
    ⊢ Eq (MulAction.toPermHom G α).range Top.top
  -/
  have := MulAction.IsPretransitive.of_compHom (α := α) (MulAction.toPermHom G α).rangeRestrict
  /-
    G : Type u_1
    α : Type u_2
    inst✝³ : Group G
    inst✝² : MulAction G α
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    S : Set G
    hS1 : ∀ (σ : G), Membership.mem S σ → ((MulAction.toPermHom G α) σ).IsSwap
    hS2 : Eq (Subgroup.closure S) Top.top
    h : MulAction.IsPretransitive G α
    this : MulAction.IsPretransitive (Subtype fun x => Membership.mem (MulAction.t …
    ⊢ Eq (MulAction.toPermHom G α).range Top.top
  -/
  rw [MonoidHom.range_eq_map, ← hS2, MonoidHom.map_closure] at this ⊢
  /-
    G : Type u_1
    α : Type u_2
    inst✝³ : Group G
    inst✝² : MulAction G α
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    S : Set G
    hS1 : ∀ (σ : G), Membership.mem S σ → ((MulAction.toPermHom G α) σ).IsSwap
    hS2 : Eq (Subgroup.closure S) Top.top
    h : MulAction.IsPretransitive G α
    this : MulAction.IsPretransitive (Subtype fun x => Membership.mem (Subgroup.cl …
    ⊢ Eq (Subgroup.closure (Set.image (⇑(MulAction.toPermHom G α)) S)) Top.top
  -/
  exact closure_of_isSwap_of_isPretransitive (Set.forall_mem_image.2 hS1)
  /-
    🎉 no goals
  -/

