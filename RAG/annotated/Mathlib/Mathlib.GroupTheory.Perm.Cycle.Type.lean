/-- The cycle type of a permutation -/
def cycleType (σ : Perm α) : Multiset ℕ :=
  σ.cycleFactorsFinset.1.map (Finset.card ∘ support)


theorem cycleType_def (σ : Perm α) :
    σ.cycleType = σ.cycleFactorsFinset.1.map (Finset.card ∘ support) :=
  rfl


theorem cycleType_eq' {σ : Perm α} (s : Finset (Perm α)) (h1 : ∀ f : Perm α, f ∈ s → f.IsCycle)
    (h2 : (s : Set (Perm α)).Pairwise Disjoint)
    (h0 : s.noncommProd id (h2.imp fun _ _ => Disjoint.commute) = σ) :
    σ.cycleType = s.1.map (Finset.card ∘ support) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    s : Finset (Equiv.Perm α)
    h1 : ∀ (f : Equiv.Perm α), Membership.mem s f → f.IsCycle
    h2 : (↑s).Pairwise Equiv.Perm.Disjoint
    h0 : Eq (s.noncommProd id ⋯) σ
    ⊢ Eq σ.cycleType (Multiset.map (Function.comp Finset.card Equiv.Perm.support)  …
  -/
  rw [cycleType_def]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    s : Finset (Equiv.Perm α)
    h1 : ∀ (f : Equiv.Perm α), Membership.mem s f → f.IsCycle
    h2 : (↑s).Pairwise Equiv.Perm.Disjoint
    h0 : Eq (s.noncommProd id ⋯) σ
    ⊢ Eq (Multiset.map (Function.comp Finset.card Equiv.Perm.support) σ.cycleFacto …
  -/
  congr
  /-
    case e_s.e_self
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    s : Finset (Equiv.Perm α)
    h1 : ∀ (f : Equiv.Perm α), Membership.mem s f → f.IsCycle
    h2 : (↑s).Pairwise Equiv.Perm.Disjoint
    h0 : Eq (s.noncommProd id ⋯) σ
    ⊢ Eq σ.cycleFactorsFinset s
  -/
  rw [cycleFactorsFinset_eq_finset]
  /-
    case e_s.e_self
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    s : Finset (Equiv.Perm α)
    h1 : ∀ (f : Equiv.Perm α), Membership.mem s f → f.IsCycle
    h2 : (↑s).Pairwise Equiv.Perm.Disjoint
    h0 : Eq (s.noncommProd id ⋯) σ
    ⊢ And (∀ (f : Equiv.Perm α), Membership.mem s f → f.IsCycle) (Exists fun h =>  …
  -/
  exact ⟨h1, h2, h0⟩
  /-
    🎉 no goals
  -/


theorem cycleType_eq {σ : Perm α} (l : List (Perm α)) (h0 : l.prod = σ)
    (h1 : ∀ σ : Perm α, σ ∈ l → σ.IsCycle) (h2 : l.Pairwise Disjoint) :
    σ.cycleType = l.map (Finset.card ∘ support) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    l : List (Equiv.Perm α)
    h0 : Eq l.prod σ
    h1 : ∀ (σ : Equiv.Perm α), Membership.mem l σ → σ.IsCycle
    h2 : List.Pairwise Equiv.Perm.Disjoint l
    ⊢ Eq σ.cycleType ↑(List.map (Function.comp Finset.card Equiv.Perm.support) l)
  -/
  have hl : l.Nodup := nodup_of_pairwise_disjoint_cycles h1 h2
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    l : List (Equiv.Perm α)
    h0 : Eq l.prod σ
    h1 : ∀ (σ : Equiv.Perm α), Membership.mem l σ → σ.IsCycle
    h2 : List.Pairwise Equiv.Perm.Disjoint l
    hl : l.Nodup
    ⊢ Eq σ.cycleType ↑(List.map (Function.comp Finset.card Equiv.Perm.support) l)
  -/
  rw [cycleType_eq' l.toFinset]
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      h0 : Eq l.prod σ
      h1 : ∀ (σ : Equiv.Perm α), Membership.mem l σ → σ.IsCycle
      h2 : List.Pairwise Equiv.Perm.Disjoint l
      hl : l.Nodup
      ⊢ Eq (Multiset.map (Function.comp Finset.card Equiv.Perm.support) l.toFinset.v …
    -/
  · simp [List.dedup_eq_self.mpr hl, Function.comp_def]
    /-
      🎉 no goals
    -/
    /-
      case h1
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      h0 : Eq l.prod σ
      h1 : ∀ (σ : Equiv.Perm α), Membership.mem l σ → σ.IsCycle
      h2 : List.Pairwise Equiv.Perm.Disjoint l
      hl : l.Nodup
      ⊢ ∀ (f : Equiv.Perm α), Membership.mem l.toFinset f → f.IsCycle
    -/
  · simpa using h1
    /-
      🎉 no goals
    -/
    /-
      case h2
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      h0 : Eq l.prod σ
      h1 : ∀ (σ : Equiv.Perm α), Membership.mem l σ → σ.IsCycle
      h2 : List.Pairwise Equiv.Perm.Disjoint l
      hl : l.Nodup
      ⊢ (↑l.toFinset).Pairwise Equiv.Perm.Disjoint
    -/
  · simpa [hl] using h2
    /-
      🎉 no goals
    -/
    /-
      case h0
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      l : List (Equiv.Perm α)
      h0 : Eq l.prod σ
      h1 : ∀ (σ : Equiv.Perm α), Membership.mem l σ → σ.IsCycle
      h2 : List.Pairwise Equiv.Perm.Disjoint l
      hl : l.Nodup
      ⊢ Eq (l.toFinset.noncommProd id ⋯) σ
    -/
  · simp [hl, h0]
    /-
      🎉 no goals
    -/


theorem CycleType.count_def {σ : Perm α} (n : ℕ) :
    σ.cycleType.count n =
      Fintype.card {c : σ.cycleFactorsFinset // (c : Perm α).support.card = n } := by
  -- work on the LHS
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    n : Nat
    ⊢ Eq (Multiset.count n σ.cycleType) (Fintype.card (Subtype fun c => Eq (↑c).su …
  -/
  rw [cycleType, Multiset.count_eq_card_filter_eq]
  -- rewrite the `Fintype.card` as a `Finset.card`
  rw [Fintype.subtype_card, Finset.univ_eq_attach, Finset.filter_attach',
    Finset.card_map, Finset.card_attach]
  simp only [Function.comp_apply, Finset.card, Finset.filter_val,
    Multiset.filter_map, Multiset.card_map]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    n : Nat
    ⊢ Eq (Multiset.filter (fun x => Eq n x.support.val.card) σ.cycleFactorsFinset. …
  -/
  congr 1
  /-
    case e_s
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    n : Nat
    ⊢ Eq (Multiset.filter (fun x => Eq n x.support.val.card) σ.cycleFactorsFinset. …
  -/
  apply Multiset.filter_congr
  /-
    case e_s.a
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    n : Nat
    ⊢ ∀ (x : Equiv.Perm α), Membership.mem σ.cycleFactorsFinset.val x → Iff (Eq n  …
  -/
  intro d h
  /-
    case e_s.a
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    n : Nat
    d : Equiv.Perm α
    h : Membership.mem σ.cycleFactorsFinset.val d
    ⊢ Iff (Eq n d.support.val.card) (Exists fun h => Eq d.support.val.card n)
  -/
  simp only [Function.comp_apply, eq_comm, Finset.mem_val.mp h, exists_const]
  /-
    🎉 no goals
  -/


@[simp] -- Porting note: new attr
theorem cycleType_eq_zero {σ : Perm α} : σ.cycleType = 0 ↔ σ = 1 := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    ⊢ Iff (Eq σ.cycleType 0) (Eq σ 1)
  -/
  simp [cycleType_def, cycleFactorsFinset_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp] -- Porting note: new attr
theorem cycleType_one : (1 : Perm α).cycleType = 0 := cycleType_eq_zero.2 rfl


theorem card_cycleType_eq_zero {σ : Perm α} : Multiset.card σ.cycleType = 0 ↔ σ = 1 := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    ⊢ Iff (Eq σ.cycleType.card 0) (Eq σ 1)
  -/
  rw [card_eq_zero, cycleType_eq_zero]
  /-
    🎉 no goals
  -/


theorem card_cycleType_pos {σ : Perm α} : 0 < Multiset.card σ.cycleType ↔ σ ≠ 1 :=
  pos_iff_ne_zero.trans card_cycleType_eq_zero.not


theorem two_le_of_mem_cycleType {σ : Perm α} {n : ℕ} (h : n ∈ σ.cycleType) : 2 ≤ n := by
  simp only [cycleType_def, ← Finset.mem_def, Function.comp_apply, Multiset.mem_map,
    mem_cycleFactorsFinset_iff] at h
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    n : Nat
    h : Exists fun a => And (And a.IsCycle (∀ (a_1 : α), Membership.mem a.support  …
    ⊢ LE.le 2 n
  -/
  obtain ⟨_, ⟨hc, -⟩, rfl⟩ := h
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ w✝ : Equiv.Perm α
    hc : w✝.IsCycle
    ⊢ LE.le 2 w✝.support.card
  -/
  exact hc.two_le_card_support
  /-
    🎉 no goals
  -/


theorem one_lt_of_mem_cycleType {σ : Perm α} {n : ℕ} (h : n ∈ σ.cycleType) : 1 < n :=
  two_le_of_mem_cycleType h


theorem IsCycle.cycleType {σ : Perm α} (hσ : IsCycle σ) : σ.cycleType = [σ.support.card] :=
  cycleType_eq [σ] (mul_one σ) (fun _τ hτ => (congr_arg IsCycle (List.mem_singleton.mp hτ)).mpr hσ)
    (List.pairwise_singleton Disjoint σ)


theorem card_cycleType_eq_one {σ : Perm α} : Multiset.card σ.cycleType = 1 ↔ σ.IsCycle := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    ⊢ Iff (Eq σ.cycleType.card 1) σ.IsCycle
  -/
  rw [card_eq_one]
  simp_rw [cycleType_def, Multiset.map_eq_singleton, ← Finset.singleton_val, Finset.val_inj,
    cycleFactorsFinset_eq_singleton_iff]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    ⊢ Iff (Exists fun a => Exists fun a_1 => And (And σ.IsCycle (Eq σ a_1)) (Eq (F …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      ⊢ (Exists fun a => Exists fun a_1 => And (And σ.IsCycle (Eq σ a_1)) (Eq (Funct …
    -/
  · rintro ⟨_, _, ⟨h, -⟩, -⟩
    /-
      case mp.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      w✝¹ : Nat
      w✝ : Equiv.Perm α
      h : σ.IsCycle
      ⊢ σ.IsCycle
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      ⊢ σ.IsCycle → Exists fun a => Exists fun a_1 => And (And σ.IsCycle (Eq σ a_1)) …
    -/
  · intro h
    /-
      case mpr
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      h : σ.IsCycle
      ⊢ Exists fun a => Exists fun a_1 => And (And σ.IsCycle (Eq σ a_1)) (Eq (Functi …
    -/
    use σ.support.card, σ
    /-
      case h
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      h : σ.IsCycle
      ⊢ And (And σ.IsCycle (Eq σ σ)) (Eq (Function.comp Finset.card Equiv.Perm.suppo …
    -/
    simp [h]
    /-
      🎉 no goals
    -/


theorem Disjoint.cycleType {σ τ : Perm α} (h : Disjoint σ τ) :
    (σ * τ).cycleType = σ.cycleType + τ.cycleType := by
  rw [cycleType_def, cycleType_def, cycleType_def, h.cycleFactorsFinset_mul_eq_union, ←
    Multiset.map_add, Finset.union_val, Multiset.add_eq_union_iff_disjoint.mpr _]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    h : σ.Disjoint τ
    ⊢ _root_.Disjoint σ.cycleFactorsFinset.val τ.cycleFactorsFinset.val
  -/
  exact Finset.disjoint_val.2 h.disjoint_cycleFactorsFinset
  /-
    🎉 no goals
  -/


@[simp] -- Porting note: new attr
theorem cycleType_inv (σ : Perm α) : σ⁻¹.cycleType = σ.cycleType :=
  cycle_induction_on (P := fun τ : Perm α => τ⁻¹.cycleType = τ.cycleType) σ rfl
                    /-
                      α : Type u_1
                      inst✝¹ : Fintype α
                      inst✝ : DecidableEq α
                      σ✝ σ : Equiv.Perm α
                      hσ : σ.IsCycle
                      ⊢ (fun τ => Eq (Inv.inv τ).cycleType τ.cycleType) σ
                    -/
    (fun σ hσ => by simp only [hσ.cycleType, hσ.inv.cycleType, support_inv])
                    /-
                      🎉 no goals
                    -/
    fun σ τ hστ _ hσ hτ => by
      simp only [mul_inv_rev, hστ.cycleType, hστ.symm.inv_left.inv_right.cycleType, hσ, hτ,
        add_comm]


@[simp] -- Porting note: new attr
theorem cycleType_conj {σ τ : Perm α} : (τ * σ * τ⁻¹).cycleType = σ.cycleType := by
  induction σ using cycle_induction_on with
  | base_one => simp
  | base_cycles σ hσ => rw [hσ.cycleType, hσ.conj.cycleType, card_support_conj]
  | induction_disjoint σ π hd _ hσ hπ =>
    rw [← conj_mul, hd.cycleType, (hd.conj _).cycleType, hσ, hπ]


theorem sum_cycleType (σ : Perm α) : σ.cycleType.sum = σ.support.card := by
  induction σ using cycle_induction_on with
  | base_one => simp
  | base_cycles σ hσ => rw [hσ.cycleType, sum_coe, List.sum_singleton]
  | induction_disjoint σ τ hd _ hσ hτ => rw [hd.cycleType, sum_add, hσ, hτ, hd.card_support_mul]


theorem card_fixedPoints (σ : Equiv.Perm α) :
    Fintype.card (Function.fixedPoints σ) = Fintype.card α - σ.cycleType.sum := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    ⊢ Eq (Fintype.card ↑(Function.fixedPoints ⇑σ)) (HSub.hSub (Fintype.card α) σ.c …
  -/
  rw [Equiv.Perm.sum_cycleType, ← Finset.card_compl, Fintype.card_ofFinset]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    ⊢ Eq (Finset.filter (Membership.mem (Function.fixedPoints ⇑σ)) Finset.univ).ca …
  -/
  congr; aesop
         /-
           🎉 no goals
         -/


theorem sign_of_cycleType' (σ : Perm α) :
    sign σ = (σ.cycleType.map fun n => -(-1 : ℤˣ) ^ n).prod := by
  induction σ using cycle_induction_on with
  | base_one => simp
  | base_cycles σ hσ => simp [hσ.cycleType, hσ.sign]
  | induction_disjoint σ τ hd _ hσ hτ => simp [hσ, hτ, hd.cycleType]


theorem sign_of_cycleType (f : Perm α) :
    sign f = (-1 : ℤˣ) ^ (f.cycleType.sum + Multiset.card f.cycleType) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    ⊢ Eq (Equiv.Perm.sign f) (HPow.hPow (-1) (HAdd.hAdd f.cycleType.sum f.cycleTyp …
  -/
  rw [sign_of_cycleType']
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    ⊢ Eq (Multiset.map (fun n => Neg.neg (HPow.hPow (-1) n)) f.cycleType).prod (HP …
  -/
  induction' f.cycleType using Multiset.induction_on with a s ihs
    /-
      case empty
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      ⊢ Eq (Multiset.map (fun n => Neg.neg (HPow.hPow (-1) n)) 0).prod (HPow.hPow (- …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      a : Nat
      s : Multiset Nat
      ihs : Eq (Multiset.map (fun n => Neg.neg (HPow.hPow (-1) n)) s).prod (HPow.hPo …
      ⊢ Eq (Multiset.map (fun n => Neg.neg (HPow.hPow (-1) n)) (Multiset.cons a s)). …
    -/
  · rw [Multiset.map_cons, Multiset.prod_cons, Multiset.sum_cons, Multiset.card_cons, ihs]
    /-
      case cons
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      a : Nat
      s : Multiset Nat
      ihs : Eq (Multiset.map (fun n => Neg.neg (HPow.hPow (-1) n)) s).prod (HPow.hPo …
      ⊢ Eq (HMul.hMul (Neg.neg (HPow.hPow (-1) a)) (HPow.hPow (-1) (HAdd.hAdd s.sum  …
    -/
    simp only [pow_add, pow_one, mul_neg_one, neg_mul, mul_neg, mul_assoc, mul_one]
    /-
      🎉 no goals
    -/


@[simp] -- Porting note: new attr
theorem lcm_cycleType (σ : Perm α) : σ.cycleType.lcm = orderOf σ := by
  induction σ using cycle_induction_on with
  | base_one => simp
  | base_cycles σ hσ => simp [hσ.cycleType, hσ.orderOf]
  | induction_disjoint σ τ hd _ hσ hτ => simp [hd.cycleType, hd.orderOf, lcm_eq_nat_lcm, hσ, hτ]


theorem dvd_of_mem_cycleType {σ : Perm α} {n : ℕ} (h : n ∈ σ.cycleType) : n ∣ orderOf σ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    n : Nat
    h : Membership.mem σ.cycleType n
    ⊢ Dvd.dvd n (orderOf σ)
  -/
  rw [← lcm_cycleType]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    n : Nat
    h : Membership.mem σ.cycleType n
    ⊢ Dvd.dvd n σ.cycleType.lcm
  -/
  exact dvd_lcm h
  /-
    🎉 no goals
  -/


theorem orderOf_cycleOf_dvd_orderOf (f : Perm α) (x : α) : orderOf (cycleOf f x) ∣ orderOf f := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    x : α
    ⊢ Dvd.dvd (orderOf (f.cycleOf x)) (orderOf f)
  -/
  by_cases hx : f x = x
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      hx : Eq (f x) x
      ⊢ Dvd.dvd (orderOf (f.cycleOf x)) (orderOf f)
    -/
  · rw [← cycleOf_eq_one_iff] at hx
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      hx : Eq (f.cycleOf x) 1
      ⊢ Dvd.dvd (orderOf (f.cycleOf x)) (orderOf f)
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      hx : Not (Eq (f x) x)
      ⊢ Dvd.dvd (orderOf (f.cycleOf x)) (orderOf f)
    -/
  · refine dvd_of_mem_cycleType ?_
    /-
      case neg
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      hx : Not (Eq (f x) x)
      ⊢ Membership.mem f.cycleType (orderOf (f.cycleOf x))
    -/
    rw [cycleType, Multiset.mem_map]
    /-
      case neg
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      hx : Not (Eq (f x) x)
      ⊢ Exists fun a => And (Membership.mem f.cycleFactorsFinset.val a) (Eq (Functio …
    -/
    refine ⟨f.cycleOf x, ?_, ?_⟩
      /-
        case neg.refine_1
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        f : Equiv.Perm α
        x : α
        hx : Not (Eq (f x) x)
        ⊢ Membership.mem f.cycleFactorsFinset.val (f.cycleOf x)
      -/
    · rwa [← Finset.mem_def, cycleOf_mem_cycleFactorsFinset_iff, mem_support]
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        f : Equiv.Perm α
        x : α
        hx : Not (Eq (f x) x)
        ⊢ Eq (Function.comp Finset.card Equiv.Perm.support (f.cycleOf x)) (orderOf (f. …
      -/
    · simp [(isCycle_cycleOf _ hx).orderOf]
      /-
        🎉 no goals
      -/


theorem two_dvd_card_support {σ : Perm α} (hσ : σ ^ 2 = 1) : 2 ∣ σ.support.card :=
  (congr_arg (Dvd.dvd 2) σ.sum_cycleType).mp
    (Multiset.dvd_sum fun n hn => by
      rw [_root_.le_antisymm
          (Nat.le_of_dvd zero_lt_two <|
            (dvd_of_mem_cycleType hn).trans <| orderOf_dvd_of_pow_eq_one hσ)
          (two_le_of_mem_cycleType hn)])


theorem cycleType_prime_order {σ : Perm α} (hσ : (orderOf σ).Prime) :
    ∃ n : ℕ, σ.cycleType = Multiset.replicate (n + 1) (orderOf σ) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    hσ : Nat.Prime (orderOf σ)
    ⊢ Exists fun n => Eq σ.cycleType (Multiset.replicate (HAdd.hAdd n 1) (orderOf  …
  -/
  refine ⟨Multiset.card σ.cycleType - 1, eq_replicate.2 ⟨?_, fun n hn ↦ ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : Nat.Prime (orderOf σ)
      ⊢ Eq σ.cycleType.card (HAdd.hAdd (HSub.hSub σ.cycleType.card 1) 1)
    -/
  · rw [tsub_add_cancel_of_le]
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : Nat.Prime (orderOf σ)
      ⊢ LE.le 1 σ.cycleType.card
    -/
    rw [Nat.succ_le_iff, card_cycleType_pos, Ne, ← orderOf_eq_one_iff]
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : Nat.Prime (orderOf σ)
      ⊢ Not (Eq (orderOf σ) 1)
    -/
    exact hσ.ne_one
    /-
      🎉 no goals
    -/
  · exact (hσ.eq_one_or_self_of_dvd n (dvd_of_mem_cycleType hn)).resolve_left
      (one_lt_of_mem_cycleType hn).ne'


theorem isCycle_of_prime_order {σ : Perm α} (h1 : (orderOf σ).Prime)
    (h2 : σ.support.card < 2 * orderOf σ) : σ.IsCycle := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h1 : Nat.Prime (orderOf σ)
    h2 : LT.lt σ.support.card (HMul.hMul 2 (orderOf σ))
    ⊢ σ.IsCycle
  -/
  obtain ⟨n, hn⟩ := cycleType_prime_order h1
  rw [← σ.sum_cycleType, hn, Multiset.sum_replicate, nsmul_eq_mul, Nat.cast_id,
    mul_lt_mul_right (orderOf_pos σ), Nat.succ_lt_succ_iff, Nat.lt_succ_iff, Nat.le_zero] at h2
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h1 : Nat.Prime (orderOf σ)
    n : Nat
    h2 : Eq n 0
    hn : Eq σ.cycleType (Multiset.replicate (HAdd.hAdd n 1) (orderOf σ))
    ⊢ σ.IsCycle
  -/
  rw [← card_cycleType_eq_one, hn, card_replicate, h2]
  /-
    🎉 no goals
  -/


theorem cycleType_le_of_mem_cycleFactorsFinset {f g : Perm α} (hf : f ∈ g.cycleFactorsFinset) :
    f.cycleType ≤ g.cycleType := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f g : Equiv.Perm α
    hf : Membership.mem g.cycleFactorsFinset f
    ⊢ LE.le f.cycleType g.cycleType
  -/
  have hf' := mem_cycleFactorsFinset_iff.1 hf
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f g : Equiv.Perm α
    hf : Membership.mem g.cycleFactorsFinset f
    hf' : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (g a))
    ⊢ LE.le f.cycleType g.cycleType
  -/
  rw [cycleType_def, cycleType_def, hf'.left.cycleFactorsFinset_eq_singleton]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f g : Equiv.Perm α
    hf : Membership.mem g.cycleFactorsFinset f
    hf' : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (g a))
    ⊢ LE.le (Multiset.map (Function.comp Finset.card Equiv.Perm.support) (Singleto …
  -/
  refine map_le_map ?_
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f g : Equiv.Perm α
    hf : Membership.mem g.cycleFactorsFinset f
    hf' : And f.IsCycle (∀ (a : α), Membership.mem f.support a → Eq (f a) (g a))
    ⊢ LE.le (Singleton.singleton f).val g.cycleFactorsFinset.val
  -/
  simpa only [Finset.singleton_val, singleton_le, Finset.mem_val] using hf
  /-
    🎉 no goals
  -/


theorem Disjoint.cycleType_mul {f g : Perm α} (h : f.Disjoint g) :
    (f * g).cycleType = f.cycleType + g.cycleType := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f g : Equiv.Perm α
    h : f.Disjoint g
    ⊢ Eq (HMul.hMul f g).cycleType (HAdd.hAdd f.cycleType g.cycleType)
  -/
  simp only [Perm.cycleType]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f g : Equiv.Perm α
    h : f.Disjoint g
    ⊢ Eq (Multiset.map (Function.comp Finset.card Equiv.Perm.support) (HMul.hMul f …
  -/
  rw [h.cycleFactorsFinset_mul_eq_union]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f g : Equiv.Perm α
    h : f.Disjoint g
    ⊢ Eq (Multiset.map (Function.comp Finset.card Equiv.Perm.support) (Union.union …
  -/
  simp only [Finset.union_val, Function.comp_apply]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f g : Equiv.Perm α
    h : f.Disjoint g
    ⊢ Eq (Multiset.map (fun x => x.support.card) (Union.union f.cycleFactorsFinset …
  -/
  rw [← Multiset.add_eq_union_iff_disjoint.mpr _, Multiset.map_add]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f g : Equiv.Perm α
    h : f.Disjoint g
    ⊢ _root_.Disjoint f.cycleFactorsFinset.val g.cycleFactorsFinset.val
  -/
  simp only [Finset.disjoint_val, Disjoint.disjoint_cycleFactorsFinset h]
  /-
    🎉 no goals
  -/


theorem Disjoint.cycleType_noncommProd {ι : Type*} {k : ι → Perm α} {s : Finset ι}
    (hs : Set.Pairwise s fun i j ↦ Disjoint (k i) (k j))
    (hs' : Set.Pairwise s fun i j ↦ Commute (k i) (k j) :=
      hs.imp (fun _ _ ↦ Perm.Disjoint.commute)) :
    (s.noncommProd k hs').cycleType = s.sum fun i ↦ (k i).cycleType := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | @insert i s hi hrec =>
    have hs' : (s : Set ι).Pairwise fun i j ↦ Disjoint (k i) (k j) :=
      hs.mono (by simp only [Finset.coe_insert, Set.subset_insert])
    rw [Finset.noncommProd_insert_of_not_mem _ _ _ _ hi, Finset.sum_insert hi]
    rw [Equiv.Perm.Disjoint.cycleType_mul, hrec hs']
    apply disjoint_noncommProd_right
    intro j hj
    apply hs _ _ (ne_of_mem_of_not_mem hj hi).symm <;>
      simp only [Finset.coe_insert, Set.mem_insert_iff, Finset.mem_coe, hj, or_true, true_or]



theorem cycleType_mul_inv_mem_cycleFactorsFinset_eq_sub
    {f g : Perm α} (hf : f ∈ g.cycleFactorsFinset) :
    (g * f⁻¹).cycleType = g.cycleType - f.cycleType :=
  add_right_cancel (b := f.cycleType) <| by
    rw [← (disjoint_mul_inv_of_mem_cycleFactorsFinset hf).cycleType, inv_mul_cancel_right,
      tsub_add_cancel_of_le (cycleType_le_of_mem_cycleFactorsFinset hf)]


theorem isConj_of_cycleType_eq {σ τ : Perm α} (h : cycleType σ = cycleType τ) : IsConj σ τ := by
  induction σ using cycle_induction_on generalizing τ with
  | base_one =>
    rw [cycleType_one, eq_comm, cycleType_eq_zero] at h
    rw [h]
  | base_cycles σ hσ =>
    have hτ := card_cycleType_eq_one.2 hσ
    rw [h, card_cycleType_eq_one] at hτ
    apply hσ.isConj hτ
    rw [hσ.cycleType, hτ.cycleType, coe_eq_coe, List.singleton_perm] at h
    exact List.singleton_injective h
  | induction_disjoint σ π hd hc hσ hπ =>
    rw [hd.cycleType] at h
    have h' : σ.support.card ∈ τ.cycleType := by
      simp [← h, hc.cycleType]
    obtain ⟨σ', hσ'l, hσ'⟩ := Multiset.mem_map.mp h'
    have key : IsConj (σ' * τ * σ'⁻¹) τ := (isConj_iff.2 ⟨σ', rfl⟩).symm
    refine IsConj.trans ?_ key
    rw [mul_assoc]
    have hs : σ.cycleType = σ'.cycleType := by
      rw [← Finset.mem_def, mem_cycleFactorsFinset_iff] at hσ'l
      rw [hc.cycleType, ← hσ', hσ'l.left.cycleType]; rfl
    refine hd.isConj_mul (hσ hs) (hπ ?_) ?_
    · rw [cycleType_mul_inv_mem_cycleFactorsFinset_eq_sub, ← h, add_comm, hs,
        add_tsub_cancel_right]
      rwa [Finset.mem_def]
    · exact (disjoint_mul_inv_of_mem_cycleFactorsFinset hσ'l).symm


theorem isConj_iff_cycleType_eq {σ τ : Perm α} : IsConj σ τ ↔ σ.cycleType = τ.cycleType :=
  ⟨fun h => by
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ τ : Equiv.Perm α
      h : IsConj σ τ
      ⊢ Eq σ.cycleType τ.cycleType
    -/
    obtain ⟨π, rfl⟩ := isConj_iff.1 h
    /-
      case intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ π : Equiv.Perm α
      h : IsConj σ (HMul.hMul (HMul.hMul π σ) (Inv.inv π))
      ⊢ Eq σ.cycleType (HMul.hMul (HMul.hMul π σ) (Inv.inv π)).cycleType
    -/
    rw [cycleType_conj], isConj_of_cycleType_eq⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem cycleType_extendDomain {β : Type*} [Fintype β] [DecidableEq β] {p : β → Prop}
    [DecidablePred p] (f : α ≃ Subtype p) {g : Perm α} :
    cycleType (g.extendDomain f) = cycleType g := by
  induction g using cycle_induction_on with
  | base_one => rw [extendDomain_one, cycleType_one, cycleType_one]
  | base_cycles σ hσ =>
    rw [(hσ.extendDomain f).cycleType, hσ.cycleType, card_support_extend_domain]
  | induction_disjoint σ τ hd _ hσ hτ =>
    rw [hd.cycleType, ← extendDomain_mul, (hd.extendDomain f).cycleType, hσ, hτ]


theorem cycleType_ofSubtype {p : α → Prop} [DecidablePred p] {g : Perm (Subtype p)} :
    cycleType (ofSubtype g) = cycleType g :=
  cycleType_extendDomain (Equiv.refl (Subtype p))


theorem mem_cycleType_iff {n : ℕ} {σ : Perm α} :
    n ∈ cycleType σ ↔ ∃ c τ, σ = c * τ ∧ Disjoint c τ ∧ IsCycle c ∧ c.support.card = n := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    n : Nat
    σ : Equiv.Perm α
    ⊢ Iff (Membership.mem σ.cycleType n) (Exists fun c => Exists fun τ => And (Eq  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      n : Nat
      σ : Equiv.Perm α
      ⊢ Membership.mem σ.cycleType n → Exists fun c => Exists fun τ => And (Eq σ (HM …
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      n : Nat
      σ : Equiv.Perm α
      h : Membership.mem σ.cycleType n
      ⊢ Exists fun c => Exists fun τ => And (Eq σ (HMul.hMul c τ)) (And (c.Disjoint  …
    -/
    obtain ⟨l, rfl, hlc, hld⟩ := truncCycleFactors σ
    /-
      case mp.mk.mk.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      n : Nat
      l : List (Equiv.Perm α)
      h : Membership.mem l.prod.cycleType n
      x✝ : Trunc (Subtype fun l_1 => And (Eq l_1.prod l.prod) (And (∀ (g : Equiv.Per …
      hlc : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsCycle
      hld : List.Pairwise Equiv.Perm.Disjoint l
      ⊢ Exists fun c => Exists fun τ => And (Eq l.prod (HMul.hMul c τ)) (And (c.Disj …
    -/
    rw [cycleType_eq _ rfl hlc hld, Multiset.mem_coe, List.mem_map] at h
    /-
      case mp.mk.mk.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      n : Nat
      l : List (Equiv.Perm α)
      h : Exists fun a => And (Membership.mem l a) (Eq (Function.comp Finset.card Eq …
      x✝ : Trunc (Subtype fun l_1 => And (Eq l_1.prod l.prod) (And (∀ (g : Equiv.Per …
      hlc : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsCycle
      hld : List.Pairwise Equiv.Perm.Disjoint l
      ⊢ Exists fun c => Exists fun τ => And (Eq l.prod (HMul.hMul c τ)) (And (c.Disj …
    -/
    obtain ⟨c, cl, rfl⟩ := h
    /-
      case mp.mk.mk.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      l : List (Equiv.Perm α)
      x✝ : Trunc (Subtype fun l_1 => And (Eq l_1.prod l.prod) (And (∀ (g : Equiv.Per …
      hlc : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsCycle
      hld : List.Pairwise Equiv.Perm.Disjoint l
      c : Equiv.Perm α
      cl : Membership.mem l c
      ⊢ Exists fun c_1 => Exists fun τ => And (Eq l.prod (HMul.hMul c_1 τ)) (And (c_ …
    -/
    rw [(List.perm_cons_erase cl).pairwise_iff @(Disjoint.symmetric)] at hld
    /-
      case mp.mk.mk.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      l : List (Equiv.Perm α)
      x✝ : Trunc (Subtype fun l_1 => And (Eq l_1.prod l.prod) (And (∀ (g : Equiv.Per …
      hlc : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsCycle
      c : Equiv.Perm α
      hld : List.Pairwise Equiv.Perm.Disjoint (List.cons c (l.erase c))
      cl : Membership.mem l c
      ⊢ Exists fun c_1 => Exists fun τ => And (Eq l.prod (HMul.hMul c_1 τ)) (And (c_ …
    -/
    refine ⟨c, (l.erase c).prod, ?_, ?_, hlc _ cl, rfl⟩
      /-
        case mp.mk.mk.intro.intro.intro.intro.refine_1
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        l : List (Equiv.Perm α)
        x✝ : Trunc (Subtype fun l_1 => And (Eq l_1.prod l.prod) (And (∀ (g : Equiv.Per …
        hlc : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsCycle
        c : Equiv.Perm α
        hld : List.Pairwise Equiv.Perm.Disjoint (List.cons c (l.erase c))
        cl : Membership.mem l c
        ⊢ Eq l.prod (HMul.hMul c (l.erase c).prod)
      -/
    · rw [← List.prod_cons, (List.perm_cons_erase cl).symm.prod_eq' (hld.imp Disjoint.commute)]
      /-
        🎉 no goals
      -/
      /-
        case mp.mk.mk.intro.intro.intro.intro.refine_2
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        l : List (Equiv.Perm α)
        x✝ : Trunc (Subtype fun l_1 => And (Eq l_1.prod l.prod) (And (∀ (g : Equiv.Per …
        hlc : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsCycle
        c : Equiv.Perm α
        hld : List.Pairwise Equiv.Perm.Disjoint (List.cons c (l.erase c))
        cl : Membership.mem l c
        ⊢ c.Disjoint (l.erase c).prod
      -/
    · exact disjoint_prod_right _ fun g => List.rel_of_pairwise_cons hld
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      n : Nat
      σ : Equiv.Perm α
      ⊢ (Exists fun c => Exists fun τ => And (Eq σ (HMul.hMul c τ)) (And (c.Disjoint …
    -/
  · rintro ⟨c, t, rfl, hd, hc, rfl⟩
    /-
      case mpr.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      c t : Equiv.Perm α
      hd : c.Disjoint t
      hc : c.IsCycle
      ⊢ Membership.mem (HMul.hMul c t).cycleType c.support.card
    -/
    simp [hd.cycleType, hc.cycleType]
    /-
      🎉 no goals
    -/


theorem le_card_support_of_mem_cycleType {n : ℕ} {σ : Perm α} (h : n ∈ cycleType σ) :
    n ≤ σ.support.card :=
  (le_sum_of_mem h).trans (le_of_eq σ.sum_cycleType)


theorem cycleType_of_card_le_mem_cycleType_add_two {n : ℕ} {g : Perm α}
    (hn2 : Fintype.card α < n + 2) (hng : n ∈ g.cycleType) : g.cycleType = {n} := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    n : Nat
    g : Equiv.Perm α
    hn2 : LT.lt (Fintype.card α) (HAdd.hAdd n 2)
    hng : Membership.mem g.cycleType n
    ⊢ Eq g.cycleType (Singleton.singleton n)
  -/
  obtain ⟨c, g', rfl, hd, hc, rfl⟩ := mem_cycleType_iff.1 hng
  suffices g'1 : g' = 1 by
    rw [hd.cycleType, hc.cycleType, coe_singleton, g'1, cycleType_one, add_zero]
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    c g' : Equiv.Perm α
    hd : c.Disjoint g'
    hc : c.IsCycle
    hn2 : LT.lt (Fintype.card α) (HAdd.hAdd c.support.card 2)
    hng : Membership.mem (HMul.hMul c g').cycleType c.support.card
    ⊢ Eq g' 1
  -/
  contrapose! hn2 with g'1
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    c g' : Equiv.Perm α
    hd : c.Disjoint g'
    hc : c.IsCycle
    hng : Membership.mem (HMul.hMul c g').cycleType c.support.card
    g'1 : Ne g' 1
    ⊢ LE.le (HAdd.hAdd c.support.card 2) (Fintype.card α)
  -/
  apply le_trans _ (c * g').support.card_le_univ
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    c g' : Equiv.Perm α
    hd : c.Disjoint g'
    hc : c.IsCycle
    hng : Membership.mem (HMul.hMul c g').cycleType c.support.card
    g'1 : Ne g' 1
    ⊢ LE.le (HAdd.hAdd c.support.card 2) (HMul.hMul c g').support.card
  -/
  rw [hd.card_support_mul]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    c g' : Equiv.Perm α
    hd : c.Disjoint g'
    hc : c.IsCycle
    hng : Membership.mem (HMul.hMul c g').cycleType c.support.card
    g'1 : Ne g' 1
    ⊢ LE.le (HAdd.hAdd c.support.card 2) (HAdd.hAdd c.support.card g'.support.card)
  -/
  exact add_le_add_left (two_le_card_support_of_ne_one g'1) _
  /-
    🎉 no goals
  -/


theorem card_compl_support_modEq [DecidableEq α] {p n : ℕ} [hp : Fact p.Prime] {σ : Perm α}
    (hσ : σ ^ p ^ n = 1) : σ.supportᶜ.card ≡ Fintype.card α [MOD p] := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p n : Nat
    hp : Fact (Nat.Prime p)
    σ : Equiv.Perm α
    hσ : Eq (HPow.hPow σ (HPow.hPow p n)) 1
    ⊢ p.ModEq (HasCompl.compl σ.support).card (Fintype.card α)
  -/
  rw [Nat.modEq_iff_dvd', ← Finset.card_compl, compl_compl, ← sum_cycleType]
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p n : Nat
      hp : Fact (Nat.Prime p)
      σ : Equiv.Perm α
      hσ : Eq (HPow.hPow σ (HPow.hPow p n)) 1
      ⊢ Dvd.dvd p σ.cycleType.sum
    -/
  · refine Multiset.dvd_sum fun k hk => ?_
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p n : Nat
      hp : Fact (Nat.Prime p)
      σ : Equiv.Perm α
      hσ : Eq (HPow.hPow σ (HPow.hPow p n)) 1
      k : Nat
      hk : Membership.mem σ.cycleType k
      ⊢ Dvd.dvd p k
    -/
    obtain ⟨m, -, hm⟩ := (Nat.dvd_prime_pow hp.out).mp (orderOf_dvd_of_pow_eq_one hσ)
    obtain ⟨l, -, rfl⟩ := (Nat.dvd_prime_pow hp.out).mp
      ((congr_arg _ hm).mp (dvd_of_mem_cycleType hk))
    /-
      case intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p n : Nat
      hp : Fact (Nat.Prime p)
      σ : Equiv.Perm α
      hσ : Eq (HPow.hPow σ (HPow.hPow p n)) 1
      m : Nat
      hm : Eq (orderOf σ) (HPow.hPow p m)
      l : Nat
      hk : Membership.mem σ.cycleType (HPow.hPow p l)
      ⊢ Dvd.dvd p (HPow.hPow p l)
    -/
    exact dvd_pow_self _ fun h => (one_lt_of_mem_cycleType hk).ne <| by rw [h, pow_zero]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p n : Nat
      hp : Fact (Nat.Prime p)
      σ : Equiv.Perm α
      hσ : Eq (HPow.hPow σ (HPow.hPow p n)) 1
      ⊢ LE.le (HasCompl.compl σ.support).card (Fintype.card α)
    -/
  · exact Finset.card_le_univ _
    /-
      🎉 no goals
    -/


open Function in
/-- The number of fixed points of a `p ^ n`-th root of the identity function over a finite set
and the set's cardinality have the same residue modulo `p`, where `p` is a prime. -/
theorem card_fixedPoints_modEq [DecidableEq α] {f : Function.End α} {p n : ℕ}
    [hp : Fact p.Prime] (hf : f ^ p ^ n = 1) :
    Fintype.card α ≡ Fintype.card f.fixedPoints [MOD p] := by
  let σ : α ≃ α := ⟨f, f ^ (p ^ n - 1),
    leftInverse_iff_comp.mpr ((pow_sub_mul_pow f (Nat.one_le_pow n p hp.out.pos)).trans hf),
    leftInverse_iff_comp.mpr ((pow_mul_pow_sub f (Nat.one_le_pow n p hp.out.pos)).trans hf)⟩
  have hσ : σ ^ p ^ n = 1 := by
    rw [DFunLike.ext'_iff, coe_pow]
    exact (hom_coe_pow (fun g : Function.End α ↦ g) rfl (fun g h ↦ rfl) f (p ^ n)).symm.trans hf
  suffices Fintype.card f.fixedPoints = (support σ)ᶜ.card from
    this ▸ (card_compl_support_modEq hσ).symm
  suffices f.fixedPoints = (support σ)ᶜ by
    simp only [this]; apply Fintype.card_coe
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Function.End α
    p n : Nat
    hp : Fact (Nat.Prime p)
    hf : Eq (HPow.hPow f (HPow.hPow p n)) 1
    σ : Equiv α α := { toFun := f, invFun := HPow.hPow f (HSub.hSub (HPow.hPow p n …
    hσ : Eq (HPow.hPow σ (HPow.hPow p n)) 1
    ⊢ Eq (Function.fixedPoints f) ↑(HasCompl.compl (Equiv.Perm.support σ))
  -/
  simp [σ, Set.ext_iff, IsFixedPt]
  /-
    🎉 no goals
  -/


theorem exists_fixed_point_of_prime {p n : ℕ} [hp : Fact p.Prime] (hα : ¬p ∣ Fintype.card α)
    {σ : Perm α} (hσ : σ ^ p ^ n = 1) : ∃ a : α, σ a = a := by
  classical
    contrapose! hα
    simp_rw [← mem_support, ← Finset.eq_univ_iff_forall] at hα
    exact Nat.modEq_zero_iff_dvd.1 ((congr_arg _ (Finset.card_eq_zero.2 (compl_eq_bot.2 hα))).mp
      (card_compl_support_modEq hσ).symm)


theorem exists_fixed_point_of_prime' {p n : ℕ} [hp : Fact p.Prime] (hα : p ∣ Fintype.card α)
    {σ : Perm α} (hσ : σ ^ p ^ n = 1) {a : α} (ha : σ a = a) : ∃ b : α, σ b = b ∧ b ≠ a := by
  classical
    have h : ∀ b : α, b ∈ σ.supportᶜ ↔ σ b = b := fun b => by
      rw [Finset.mem_compl, mem_support, Classical.not_not]
    obtain ⟨b, hb1, hb2⟩ := Finset.exists_ne_of_one_lt_card (hp.out.one_lt.trans_le
      (Nat.le_of_dvd (Finset.card_pos.mpr ⟨a, (h a).mpr ha⟩) (Nat.modEq_zero_iff_dvd.mp
        ((card_compl_support_modEq hσ).trans (Nat.modEq_zero_iff_dvd.mpr hα))))) a
    exact ⟨b, (h b).mp hb1, hb2⟩


theorem isCycle_of_prime_order' {σ : Perm α} (h1 : (orderOf σ).Prime)
    (h2 : Fintype.card α < 2 * orderOf σ) : σ.IsCycle := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    σ : Equiv.Perm α
    h1 : Nat.Prime (orderOf σ)
    h2 : LT.lt (Fintype.card α) (HMul.hMul 2 (orderOf σ))
    ⊢ σ.IsCycle
  -/
  classical exact isCycle_of_prime_order h1 (lt_of_le_of_lt σ.support.card_le_univ h2)
  /-
    🎉 no goals
  -/


theorem isCycle_of_prime_order'' {σ : Perm α} (h1 : (Fintype.card α).Prime)
    (h2 : orderOf σ = Fintype.card α) : σ.IsCycle :=
  isCycle_of_prime_order' ((congr_arg Nat.Prime h2).mpr h1) <| by
    /-
      α : Type u_1
      inst✝ : Fintype α
      σ : Equiv.Perm α
      h1 : Nat.Prime (Fintype.card α)
      h2 : Eq (orderOf σ) (Fintype.card α)
      ⊢ LT.lt (Fintype.card α) (HMul.hMul 2 (orderOf σ))
    -/
    rw [← one_mul (Fintype.card α), ← h2, mul_lt_mul_right (orderOf_pos σ)]
    /-
      α : Type u_1
      inst✝ : Fintype α
      σ : Equiv.Perm α
      h1 : Nat.Prime (Fintype.card α)
      h2 : Eq (orderOf σ) (Fintype.card α)
      ⊢ LT.lt 1 2
    -/
    exact one_lt_two
    /-
      🎉 no goals
    -/


/-- The type of vectors with terms from `G`, length `n`, and product equal to `1:G`. -/
def vectorsProdEqOne : Set (List.Vector G n) :=
  { v | v.toList.prod = 1 }


theorem mem_iff {n : ℕ} (v : List.Vector G n) : v ∈ vectorsProdEqOne G n ↔ v.toList.prod = 1 :=
  Iff.rfl


theorem zero_eq : vectorsProdEqOne G 0 = {Vector.nil} :=
  Set.eq_singleton_iff_unique_mem.mpr ⟨Eq.refl (1 : G), fun v _ => v.eq_nil⟩


theorem one_eq : vectorsProdEqOne G 1 = {Vector.nil.cons 1} := by
  simp_rw [Set.eq_singleton_iff_unique_mem, mem_iff, Vector.toList_singleton, List.prod_singleton,
    Vector.head_cons, true_and]
  /-
    G : Type u_2
    inst✝ : Group G
    ⊢ ∀ (x : List.Vector G 1), Eq x.head 1 → Eq x (List.Vector.cons 1 List.Vector. …
  -/
  exact fun v hv => v.cons_head_tail.symm.trans (congr_arg₂ Vector.cons hv v.tail.eq_nil)
  /-
    🎉 no goals
  -/


instance zeroUnique : Unique (vectorsProdEqOne G 0) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    G : Type u_2
    inst✝ : Group G
    n : Nat
    ⊢ Unique ↑(Equiv.Perm.vectorsProdEqOne G 0)
  -/
  rw [zero_eq]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    G : Type u_2
    inst✝ : Group G
    n : Nat
    ⊢ Unique ↑(Singleton.singleton List.Vector.nil)
  -/
  exact Set.uniqueSingleton Vector.nil
  /-
    🎉 no goals
  -/


instance oneUnique : Unique (vectorsProdEqOne G 1) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    G : Type u_2
    inst✝ : Group G
    n : Nat
    ⊢ Unique ↑(Equiv.Perm.vectorsProdEqOne G 1)
  -/
  rw [one_eq]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    G : Type u_2
    inst✝ : Group G
    n : Nat
    ⊢ Unique ↑(Singleton.singleton (List.Vector.cons 1 List.Vector.nil))
  -/
  exact Set.uniqueSingleton (Vector.nil.cons 1)
  /-
    🎉 no goals
  -/


/-- Given a vector `v` of length `n`, make a vector of length `n + 1` whose product is `1`,
by appending the inverse of the product of `v`. -/
@[simps]
def vectorEquiv : List.Vector G n ≃ vectorsProdEqOne G (n + 1) where
  toFun v := ⟨v.toList.prod⁻¹ ::ᵥ v, by
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      G : Type u_2
      inst✝ : Group G
      n : Nat
      v : List.Vector G n
      ⊢ Membership.mem (Equiv.Perm.vectorsProdEqOne G (HAdd.hAdd n 1)) (List.Vector. …
    -/
    rw [mem_iff, Vector.toList_cons, List.prod_cons, inv_mul_cancel]⟩
    /-
      🎉 no goals
    -/
  invFun v := v.1.tail
  left_inv v := v.tail_cons v.toList.prod⁻¹
  right_inv v := Subtype.ext <|
    calc
      v.1.tail.toList.prod⁻¹ ::ᵥ v.1.tail = v.1.head ::ᵥ v.1.tail :=
        congr_arg (· ::ᵥ v.1.tail) <| Eq.symm <| eq_inv_of_mul_eq_one_left <| by
          /-
            α : Type u_1
            inst✝¹ : Fintype α
            G : Type u_2
            inst✝ : Group G
            n : Nat
            v : ↑(Equiv.Perm.vectorsProdEqOne G (HAdd.hAdd n 1))
            ⊢ Eq (HMul.hMul (↑v).head (↑v).tail.toList.prod) 1
          -/
          rw [← List.prod_cons, ← Vector.toList_cons, v.1.cons_head_tail]
          /-
            α : Type u_1
            inst✝¹ : Fintype α
            G : Type u_2
            inst✝ : Group G
            n : Nat
            v : ↑(Equiv.Perm.vectorsProdEqOne G (HAdd.hAdd n 1))
            ⊢ Eq (↑v).toList.prod 1
          -/
          exact v.2
          /-
            🎉 no goals
          -/
      _ = v.1 := v.1.cons_head_tail


/-- Given a vector `v` of length `n` whose product is 1, make a vector of length `n - 1`,
by deleting the last entry of `v`. -/
def equivVector : ∀ n, vectorsProdEqOne G n ≃ List.Vector G (n - 1)
  | 0 => (ofUnique (vectorsProdEqOne G 0) (vectorsProdEqOne G 1)).trans (vectorEquiv G 0).symm
  | (n + 1) => (vectorEquiv G n).symm


instance [Fintype G] : Fintype (vectorsProdEqOne G n) :=
  Fintype.ofEquiv (List.Vector G (n - 1)) (equivVector G n).symm


theorem card [Fintype G] : Fintype.card (vectorsProdEqOne G n) = Fintype.card G ^ (n - 1) :=
  (Fintype.card_congr (equivVector G n)).trans (card_vector (n - 1))


/-- Rotate a vector whose product is 1. -/
def rotate : vectorsProdEqOne G n :=
  ⟨⟨_, (v.1.1.length_rotate k).trans v.1.2⟩, List.prod_rotate_eq_one_of_prod_eq_one v.2 k⟩


theorem rotate_zero : rotate v 0 = v :=
  Subtype.ext (Subtype.ext v.1.1.rotate_zero)


theorem rotate_rotate : rotate (rotate v j) k = rotate v (j + k) :=
  Subtype.ext (Subtype.ext (v.1.1.rotate_rotate j k))


theorem rotate_length : rotate v n = v :=
  Subtype.ext (Subtype.ext ((congr_arg _ v.1.2.symm).trans v.1.1.rotate_length))


/-- For every prime `p` dividing the order of a finite group `G` there exists an element of order
`p` in `G`. This is known as Cauchy's theorem. -/
theorem _root_.exists_prime_orderOf_dvd_card {G : Type*} [Group G] [Fintype G] (p : ℕ)
    [hp : Fact p.Prime] (hdvd : p ∣ Fintype.card G) : ∃ x : G, orderOf x = p := by
  /-
    G : Type u_3
    inst✝¹ : Group G
    inst✝ : Fintype G
    p : Nat
    hp : Fact (Nat.Prime p)
    hdvd : Dvd.dvd p (Fintype.card G)
    ⊢ Exists fun x => Eq (orderOf x) p
  -/
  have hp' : p - 1 ≠ 0 := mt tsub_eq_zero_iff_le.mp (not_le_of_lt hp.out.one_lt)
  have Scard :=
    calc
      p ∣ Fintype.card G ^ (p - 1) := hdvd.trans (dvd_pow (dvd_refl _) hp')
      _ = Fintype.card (vectorsProdEqOne G p) := (VectorsProdEqOne.card G p).symm
  let f : ℕ → vectorsProdEqOne G p → vectorsProdEqOne G p := fun k v =>
    VectorsProdEqOne.rotate v k
  /-
    G : Type u_3
    inst✝¹ : Group G
    inst✝ : Fintype G
    p : Nat
    hp : Fact (Nat.Prime p)
    hdvd : Dvd.dvd p (Fintype.card G)
    hp' : Ne (HSub.hSub p 1) 0
    Scard : Dvd.dvd p (Fintype.card ↑(Equiv.Perm.vectorsProdEqOne G p))
    f : Nat → ↑(Equiv.Perm.vectorsProdEqOne G p) → ↑(Equiv.Perm.vectorsProdEqOne G …
    ⊢ Exists fun x => Eq (orderOf x) p
  -/
  have hf1 : ∀ v, f 0 v = v := VectorsProdEqOne.rotate_zero
  have hf2 : ∀ j k v, f k (f j v) = f (j + k) v := fun j k v =>
    VectorsProdEqOne.rotate_rotate v j k
  /-
    G : Type u_3
    inst✝¹ : Group G
    inst✝ : Fintype G
    p : Nat
    hp : Fact (Nat.Prime p)
    hdvd : Dvd.dvd p (Fintype.card G)
    hp' : Ne (HSub.hSub p 1) 0
    Scard : Dvd.dvd p (Fintype.card ↑(Equiv.Perm.vectorsProdEqOne G p))
    f : Nat → ↑(Equiv.Perm.vectorsProdEqOne G p) → ↑(Equiv.Perm.vectorsProdEqOne G …
    hf1 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f 0 v) v
    hf2 : ∀ (j k : Nat) (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f k (f j v)) …
    ⊢ Exists fun x => Eq (orderOf x) p
  -/
  have hf3 : ∀ v, f p v = v := VectorsProdEqOne.rotate_length
  let σ :=
    Equiv.mk (f 1) (f (p - 1)) (fun s => by rw [hf2, add_tsub_cancel_of_le hp.out.one_lt.le, hf3])
      fun s => by rw [hf2, tsub_add_cancel_of_le hp.out.one_lt.le, hf3]
  have hσ : ∀ k v, (σ ^ k) v = f k v := fun k =>
    Nat.rec (fun v => (hf1 v).symm) (fun k hk v => by
      rw [pow_succ, Perm.mul_apply, hk (σ v), Nat.succ_eq_one_add, ← hf2 1 k]
      simp only [σ, coe_fn_mk]) k
  /-
    G : Type u_3
    inst✝¹ : Group G
    inst✝ : Fintype G
    p : Nat
    hp : Fact (Nat.Prime p)
    hdvd : Dvd.dvd p (Fintype.card G)
    hp' : Ne (HSub.hSub p 1) 0
    Scard : Dvd.dvd p (Fintype.card ↑(Equiv.Perm.vectorsProdEqOne G p))
    f : Nat → ↑(Equiv.Perm.vectorsProdEqOne G p) → ↑(Equiv.Perm.vectorsProdEqOne G …
    hf1 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f 0 v) v
    hf2 : ∀ (j k : Nat) (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f k (f j v)) …
    hf3 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f p v) v
    σ : Equiv ↑(Equiv.Perm.vectorsProdEqOne G p) ↑(Equiv.Perm.vectorsProdEqOne G p …
    hσ : ∀ (k : Nat) (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq ((HPow.hPow σ k) …
    ⊢ Exists fun x => Eq (orderOf x) p
  -/
  replace hσ : σ ^ p ^ 1 = 1 := Perm.ext fun v => by rw [pow_one, hσ, hf3, one_apply]
  let v₀ : vectorsProdEqOne G p :=
    ⟨Vector.replicate p 1, (List.prod_replicate p 1).trans (one_pow p)⟩
  /-
    G : Type u_3
    inst✝¹ : Group G
    inst✝ : Fintype G
    p : Nat
    hp : Fact (Nat.Prime p)
    hdvd : Dvd.dvd p (Fintype.card G)
    hp' : Ne (HSub.hSub p 1) 0
    Scard : Dvd.dvd p (Fintype.card ↑(Equiv.Perm.vectorsProdEqOne G p))
    f : Nat → ↑(Equiv.Perm.vectorsProdEqOne G p) → ↑(Equiv.Perm.vectorsProdEqOne G …
    hf1 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f 0 v) v
    hf2 : ∀ (j k : Nat) (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f k (f j v)) …
    hf3 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f p v) v
    σ : Equiv ↑(Equiv.Perm.vectorsProdEqOne G p) ↑(Equiv.Perm.vectorsProdEqOne G p …
    hσ : Eq (HPow.hPow σ (HPow.hPow p 1)) 1
    v₀ : ↑(Equiv.Perm.vectorsProdEqOne G p) := ⟨List.Vector.replicate p 1, ⋯⟩
    ⊢ Exists fun x => Eq (orderOf x) p
  -/
  have hv₀ : σ v₀ = v₀ := Subtype.ext (Subtype.ext (List.rotate_replicate (1 : G) p 1))
  /-
    G : Type u_3
    inst✝¹ : Group G
    inst✝ : Fintype G
    p : Nat
    hp : Fact (Nat.Prime p)
    hdvd : Dvd.dvd p (Fintype.card G)
    hp' : Ne (HSub.hSub p 1) 0
    Scard : Dvd.dvd p (Fintype.card ↑(Equiv.Perm.vectorsProdEqOne G p))
    f : Nat → ↑(Equiv.Perm.vectorsProdEqOne G p) → ↑(Equiv.Perm.vectorsProdEqOne G …
    hf1 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f 0 v) v
    hf2 : ∀ (j k : Nat) (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f k (f j v)) …
    hf3 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f p v) v
    σ : Equiv ↑(Equiv.Perm.vectorsProdEqOne G p) ↑(Equiv.Perm.vectorsProdEqOne G p …
    hσ : Eq (HPow.hPow σ (HPow.hPow p 1)) 1
    v₀ : ↑(Equiv.Perm.vectorsProdEqOne G p) := ⟨List.Vector.replicate p 1, ⋯⟩
    hv₀ : Eq (σ v₀) v₀
    ⊢ Exists fun x => Eq (orderOf x) p
  -/
  obtain ⟨v, hv1, hv2⟩ := exists_fixed_point_of_prime' Scard hσ hv₀
  refine
    Exists.imp (fun g hg => orderOf_eq_prime ?_ fun hg' => hv2 ?_)
      (List.rotate_one_eq_self_iff_eq_replicate.mp (Subtype.ext_iff.mp (Subtype.ext_iff.mp hv1)))
    /-
      case intro.intro.refine_1
      G : Type u_3
      inst✝¹ : Group G
      inst✝ : Fintype G
      p : Nat
      hp : Fact (Nat.Prime p)
      hdvd : Dvd.dvd p (Fintype.card G)
      hp' : Ne (HSub.hSub p 1) 0
      Scard : Dvd.dvd p (Fintype.card ↑(Equiv.Perm.vectorsProdEqOne G p))
      f : Nat → ↑(Equiv.Perm.vectorsProdEqOne G p) → ↑(Equiv.Perm.vectorsProdEqOne G …
      hf1 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f 0 v) v
      hf2 : ∀ (j k : Nat) (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f k (f j v)) …
      hf3 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f p v) v
      σ : Equiv ↑(Equiv.Perm.vectorsProdEqOne G p) ↑(Equiv.Perm.vectorsProdEqOne G p …
      hσ : Eq (HPow.hPow σ (HPow.hPow p 1)) 1
      v₀ : ↑(Equiv.Perm.vectorsProdEqOne G p) := ⟨List.Vector.replicate p 1, ⋯⟩
      hv₀ : Eq (σ v₀) v₀
      v : ↑(Equiv.Perm.vectorsProdEqOne G p)
      hv1 : Eq (σ v) v
      hv2 : Ne v v₀
      g : G
      hg : Eq (↑↑v) (List.replicate (↑↑v).length g)
      ⊢ Eq (HPow.hPow g p) 1
    -/
  · rw [← List.prod_replicate, ← v.1.2, ← hg, show v.val.val.prod = 1 from v.2]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      G : Type u_3
      inst✝¹ : Group G
      inst✝ : Fintype G
      p : Nat
      hp : Fact (Nat.Prime p)
      hdvd : Dvd.dvd p (Fintype.card G)
      hp' : Ne (HSub.hSub p 1) 0
      Scard : Dvd.dvd p (Fintype.card ↑(Equiv.Perm.vectorsProdEqOne G p))
      f : Nat → ↑(Equiv.Perm.vectorsProdEqOne G p) → ↑(Equiv.Perm.vectorsProdEqOne G …
      hf1 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f 0 v) v
      hf2 : ∀ (j k : Nat) (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f k (f j v)) …
      hf3 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f p v) v
      σ : Equiv ↑(Equiv.Perm.vectorsProdEqOne G p) ↑(Equiv.Perm.vectorsProdEqOne G p …
      hσ : Eq (HPow.hPow σ (HPow.hPow p 1)) 1
      v₀ : ↑(Equiv.Perm.vectorsProdEqOne G p) := ⟨List.Vector.replicate p 1, ⋯⟩
      hv₀ : Eq (σ v₀) v₀
      v : ↑(Equiv.Perm.vectorsProdEqOne G p)
      hv1 : Eq (σ v) v
      hv2 : Ne v v₀
      g : G
      hg : Eq (↑↑v) (List.replicate (↑↑v).length g)
      hg' : Eq g 1
      ⊢ Eq v v₀
    -/
  · rw [Subtype.ext_iff_val, Subtype.ext_iff_val, hg, hg', v.1.2]
    /-
      case intro.intro.refine_2
      G : Type u_3
      inst✝¹ : Group G
      inst✝ : Fintype G
      p : Nat
      hp : Fact (Nat.Prime p)
      hdvd : Dvd.dvd p (Fintype.card G)
      hp' : Ne (HSub.hSub p 1) 0
      Scard : Dvd.dvd p (Fintype.card ↑(Equiv.Perm.vectorsProdEqOne G p))
      f : Nat → ↑(Equiv.Perm.vectorsProdEqOne G p) → ↑(Equiv.Perm.vectorsProdEqOne G …
      hf1 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f 0 v) v
      hf2 : ∀ (j k : Nat) (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f k (f j v)) …
      hf3 : ∀ (v : ↑(Equiv.Perm.vectorsProdEqOne G p)), Eq (f p v) v
      σ : Equiv ↑(Equiv.Perm.vectorsProdEqOne G p) ↑(Equiv.Perm.vectorsProdEqOne G p …
      hσ : Eq (HPow.hPow σ (HPow.hPow p 1)) 1
      v₀ : ↑(Equiv.Perm.vectorsProdEqOne G p) := ⟨List.Vector.replicate p 1, ⋯⟩
      hv₀ : Eq (σ v₀) v₀
      v : ↑(Equiv.Perm.vectorsProdEqOne G p)
      hv1 : Eq (σ v) v
      hv2 : Ne v v₀
      g : G
      hg : Eq (↑↑v) (List.replicate (↑↑v).length g)
      hg' : Eq g 1
      ⊢ Eq (List.replicate p 1) ↑↑v₀
    -/
    simp only [v₀, Vector.replicate]
    /-
      🎉 no goals
    -/

-- TODO: Make the `Finite` version of this theorem the default

/-- For every prime `p` dividing the order of a finite additive group `G` there exists an element of
order `p` in `G`. This is the additive version of Cauchy's theorem. -/
theorem _root_.exists_prime_addOrderOf_dvd_card {G : Type*} [AddGroup G] [Fintype G] (p : ℕ)
    [Fact p.Prime] (hdvd : p ∣ Fintype.card G) : ∃ x : G, addOrderOf x = p :=
                                                                /-
                                                                  G : Type u_3
                                                                  inst✝² : AddGroup G
                                                                  inst✝¹ : Fintype G
                                                                  p : Nat
                                                                  inst✝ : Fact (Nat.Prime p)
                                                                  hdvd : Dvd.dvd p (Fintype.card G)
                                                                  ⊢ Dvd.dvd p (Fintype.card (Multiplicative G))
                                                                -/
  @exists_prime_orderOf_dvd_card (Multiplicative G) _ _ _ _ (by convert hdvd)
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- For every prime `p` dividing the order of a finite group `G` there exists an element of order
`p` in `G`. This is known as Cauchy's theorem. -/
@[to_additive]
theorem _root_.exists_prime_orderOf_dvd_card' {G : Type*} [Group G] [Finite G] (p : ℕ)
    [hp : Fact p.Prime] (hdvd : p ∣ Nat.card G) : ∃ x : G, orderOf x = p := by
  /-
    G : Type u_3
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    hp : Fact (Nat.Prime p)
    hdvd : Dvd.dvd p (Nat.card G)
    ⊢ Exists fun x => Eq (orderOf x) p
  -/
  have := Fintype.ofFinite G
  /-
    G : Type u_3
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    hp : Fact (Nat.Prime p)
    hdvd : Dvd.dvd p (Nat.card G)
    this : Fintype G
    ⊢ Exists fun x => Eq (orderOf x) p
  -/
  rw [Nat.card_eq_fintype_card] at hdvd
  /-
    G : Type u_3
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    hp : Fact (Nat.Prime p)
    this : Fintype G
    hdvd : Dvd.dvd p (Fintype.card G)
    ⊢ Exists fun x => Eq (orderOf x) p
  -/
  exact exists_prime_orderOf_dvd_card p hdvd
  /-
    🎉 no goals
  -/


theorem subgroup_eq_top_of_swap_mem [DecidableEq α] {H : Subgroup (Perm α)}
    [d : DecidablePred (· ∈ H)] {τ : Perm α} (h0 : (Fintype.card α).Prime)
    (h1 : Fintype.card α ∣ Fintype.card H) (h2 : τ ∈ H) (h3 : IsSwap τ) : H = ⊤ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    H : Subgroup (Equiv.Perm α)
    d : DecidablePred fun x => Membership.mem H x
    τ : Equiv.Perm α
    h0 : Nat.Prime (Fintype.card α)
    h1 : Dvd.dvd (Fintype.card α) (Fintype.card (Subtype fun x => Membership.mem H …
    h2 : Membership.mem H τ
    h3 : τ.IsSwap
    ⊢ Eq H Top.top
  -/
  haveI : Fact (Fintype.card α).Prime := ⟨h0⟩
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    H : Subgroup (Equiv.Perm α)
    d : DecidablePred fun x => Membership.mem H x
    τ : Equiv.Perm α
    h0 : Nat.Prime (Fintype.card α)
    h1 : Dvd.dvd (Fintype.card α) (Fintype.card (Subtype fun x => Membership.mem H …
    h2 : Membership.mem H τ
    h3 : τ.IsSwap
    this : Fact (Nat.Prime (Fintype.card α))
    ⊢ Eq H Top.top
  -/
  obtain ⟨σ, hσ⟩ := exists_prime_orderOf_dvd_card (Fintype.card α) h1
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    H : Subgroup (Equiv.Perm α)
    d : DecidablePred fun x => Membership.mem H x
    τ : Equiv.Perm α
    h0 : Nat.Prime (Fintype.card α)
    h1 : Dvd.dvd (Fintype.card α) (Fintype.card (Subtype fun x => Membership.mem H …
    h2 : Membership.mem H τ
    h3 : τ.IsSwap
    this : Fact (Nat.Prime (Fintype.card α))
    σ : Subtype fun x => Membership.mem H x
    hσ : Eq (orderOf σ) (Fintype.card α)
    ⊢ Eq H Top.top
  -/
  have hσ1 : orderOf (σ : Perm α) = Fintype.card α := (Subgroup.orderOf_coe σ).trans hσ
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    H : Subgroup (Equiv.Perm α)
    d : DecidablePred fun x => Membership.mem H x
    τ : Equiv.Perm α
    h0 : Nat.Prime (Fintype.card α)
    h1 : Dvd.dvd (Fintype.card α) (Fintype.card (Subtype fun x => Membership.mem H …
    h2 : Membership.mem H τ
    h3 : τ.IsSwap
    this : Fact (Nat.Prime (Fintype.card α))
    σ : Subtype fun x => Membership.mem H x
    hσ : Eq (orderOf σ) (Fintype.card α)
    hσ1 : Eq (orderOf ↑σ) (Fintype.card α)
    ⊢ Eq H Top.top
  -/
  have hσ2 : IsCycle ↑σ := isCycle_of_prime_order'' h0 hσ1
  have hσ3 : (σ : Perm α).support = ⊤ :=
    Finset.eq_univ_of_card (σ : Perm α).support (hσ2.orderOf.symm.trans hσ1)
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    H : Subgroup (Equiv.Perm α)
    d : DecidablePred fun x => Membership.mem H x
    τ : Equiv.Perm α
    h0 : Nat.Prime (Fintype.card α)
    h1 : Dvd.dvd (Fintype.card α) (Fintype.card (Subtype fun x => Membership.mem H …
    h2 : Membership.mem H τ
    h3 : τ.IsSwap
    this : Fact (Nat.Prime (Fintype.card α))
    σ : Subtype fun x => Membership.mem H x
    hσ : Eq (orderOf σ) (Fintype.card α)
    hσ1 : Eq (orderOf ↑σ) (Fintype.card α)
    hσ2 : (↑σ).IsCycle
    hσ3 : Eq (↑σ).support Top.top
    ⊢ Eq H Top.top
  -/
  have hσ4 : Subgroup.closure {↑σ, τ} = ⊤ := closure_prime_cycle_swap h0 hσ2 hσ3 h3
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    H : Subgroup (Equiv.Perm α)
    d : DecidablePred fun x => Membership.mem H x
    τ : Equiv.Perm α
    h0 : Nat.Prime (Fintype.card α)
    h1 : Dvd.dvd (Fintype.card α) (Fintype.card (Subtype fun x => Membership.mem H …
    h2 : Membership.mem H τ
    h3 : τ.IsSwap
    this : Fact (Nat.Prime (Fintype.card α))
    σ : Subtype fun x => Membership.mem H x
    hσ : Eq (orderOf σ) (Fintype.card α)
    hσ1 : Eq (orderOf ↑σ) (Fintype.card α)
    hσ2 : (↑σ).IsCycle
    hσ3 : Eq (↑σ).support Top.top
    hσ4 : Eq (Subgroup.closure (Insert.insert (↑σ) (Singleton.singleton τ))) Top.top
    ⊢ Eq H Top.top
  -/
  rw [eq_top_iff, ← hσ4, Subgroup.closure_le, Set.insert_subset_iff, Set.singleton_subset_iff]
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    H : Subgroup (Equiv.Perm α)
    d : DecidablePred fun x => Membership.mem H x
    τ : Equiv.Perm α
    h0 : Nat.Prime (Fintype.card α)
    h1 : Dvd.dvd (Fintype.card α) (Fintype.card (Subtype fun x => Membership.mem H …
    h2 : Membership.mem H τ
    h3 : τ.IsSwap
    this : Fact (Nat.Prime (Fintype.card α))
    σ : Subtype fun x => Membership.mem H x
    hσ : Eq (orderOf σ) (Fintype.card α)
    hσ1 : Eq (orderOf ↑σ) (Fintype.card α)
    hσ2 : (↑σ).IsCycle
    hσ3 : Eq (↑σ).support Top.top
    hσ4 : Eq (Subgroup.closure (Insert.insert (↑σ) (Singleton.singleton τ))) Top.top
    ⊢ And (Membership.mem ↑H ↑σ) (Membership.mem (↑H) τ)
  -/
  exact ⟨Subtype.mem σ, h2⟩
  /-
    🎉 no goals
  -/


/-- The partition corresponding to a permutation -/
def partition (σ : Perm α) : (Fintype.card α).Partition where
  parts := σ.cycleType + Multiset.replicate (Fintype.card α - σ.support.card) 1
  parts_pos {n hn} := by
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      n : Nat
      hn : Membership.mem (HAdd.hAdd σ.cycleType (Multiset.replicate (HSub.hSub (Fin …
      ⊢ LT.lt 0 n
    -/
    cases' mem_add.mp hn with hn hn
      /-
        case inl
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        σ : Equiv.Perm α
        n : Nat
        hn✝ : Membership.mem (HAdd.hAdd σ.cycleType (Multiset.replicate (HSub.hSub (Fi …
        hn : Membership.mem σ.cycleType n
        ⊢ LT.lt 0 n
      -/
    · exact zero_lt_one.trans (one_lt_of_mem_cycleType hn)
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        σ : Equiv.Perm α
        n : Nat
        hn✝ : Membership.mem (HAdd.hAdd σ.cycleType (Multiset.replicate (HSub.hSub (Fi …
        hn : Membership.mem (Multiset.replicate (HSub.hSub (Fintype.card α) σ.support. …
        ⊢ LT.lt 0 n
      -/
    · exact lt_of_lt_of_le zero_lt_one (ge_of_eq (Multiset.eq_of_mem_replicate hn))
      /-
        🎉 no goals
      -/
  parts_sum := by
    rw [sum_add, sum_cycleType, Multiset.sum_replicate, nsmul_eq_mul, Nat.cast_id, mul_one,
      add_tsub_cancel_of_le σ.support.card_le_univ]


theorem parts_partition {σ : Perm α} :
    σ.partition.parts = σ.cycleType + Multiset.replicate (Fintype.card α - σ.support.card) 1 :=
  rfl


theorem filter_parts_partition_eq_cycleType {σ : Perm α} :
    ((partition σ).parts.filter fun n => 2 ≤ n) = σ.cycleType := by
  rw [parts_partition, filter_add, Multiset.filter_eq_self.2 fun _ => two_le_of_mem_cycleType,
    Multiset.filter_eq_nil.2 fun a h => ?_, add_zero]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    a : Nat
    h : Membership.mem (Multiset.replicate (HSub.hSub (Fintype.card α) σ.support.c …
    ⊢ Not (LE.le 2 a)
  -/
  rw [Multiset.eq_of_mem_replicate h]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    a : Nat
    h : Membership.mem (Multiset.replicate (HSub.hSub (Fintype.card α) σ.support.c …
    ⊢ Not (LE.le 2 1)
  -/
  decide
  /-
    🎉 no goals
  -/


theorem partition_eq_of_isConj {σ τ : Perm α} : IsConj σ τ ↔ σ.partition = τ.partition := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    ⊢ Iff (IsConj σ τ) (Eq σ.partition τ.partition)
  -/
  rw [isConj_iff_cycleType_eq]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    ⊢ Iff (Eq σ.cycleType τ.cycleType) (Eq σ.partition τ.partition)
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
  · rw [Nat.Partition.ext_iff, parts_partition, parts_partition, ← sum_cycleType, ← sum_cycleType,
      h]
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ τ : Equiv.Perm α
      h : Eq σ.partition τ.partition
      ⊢ Eq σ.cycleType τ.cycleType
    -/
  · rw [← filter_parts_partition_eq_cycleType, ← filter_parts_partition_eq_cycleType, h]
    /-
      🎉 no goals
    -/


/-- A three-cycle is a cycle of length 3. -/
def IsThreeCycle [DecidableEq α] (σ : Perm α) : Prop :=
  σ.cycleType = {3}


theorem cycleType (h : IsThreeCycle σ) : σ.cycleType = {3} :=
  h


theorem card_support (h : IsThreeCycle σ) : σ.support.card = 3 := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h : σ.IsThreeCycle
    ⊢ Eq σ.support.card 3
  -/
  rw [← sum_cycleType, h.cycleType, Multiset.sum_singleton]
  /-
    🎉 no goals
  -/


theorem _root_.card_support_eq_three_iff : σ.support.card = 3 ↔ σ.IsThreeCycle := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    ⊢ Iff (Eq σ.support.card 3) σ.IsThreeCycle
  -/
  refine ⟨fun h => ?_, IsThreeCycle.card_support⟩
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h : Eq σ.support.card 3
    ⊢ σ.IsThreeCycle
  -/
  by_cases h0 : σ.cycleType = 0
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      h : Eq σ.support.card 3
      h0 : Eq σ.cycleType 0
      ⊢ σ.IsThreeCycle
    -/
  · rw [← sum_cycleType, h0, sum_zero] at h
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      h : Eq 0 3
      h0 : Eq σ.cycleType 0
      ⊢ σ.IsThreeCycle
    -/
    exact (ne_of_lt zero_lt_three h).elim
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h : Eq σ.support.card 3
    h0 : Not (Eq σ.cycleType 0)
    ⊢ σ.IsThreeCycle
  -/
  obtain ⟨n, hn⟩ := exists_mem_of_ne_zero h0
  /-
    case neg.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h : Eq σ.support.card 3
    h0 : Not (Eq σ.cycleType 0)
    n : Nat
    hn : Membership.mem σ.cycleType n
    ⊢ σ.IsThreeCycle
  -/
  by_cases h1 : σ.cycleType.erase n = 0
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      h : Eq σ.support.card 3
      h0 : Not (Eq σ.cycleType 0)
      n : Nat
      hn : Membership.mem σ.cycleType n
      h1 : Eq (σ.cycleType.erase n) 0
      ⊢ σ.IsThreeCycle
    -/
  · rw [← sum_cycleType, ← cons_erase hn, h1, cons_zero, Multiset.sum_singleton] at h
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      h0 : Not (Eq σ.cycleType 0)
      n : Nat
      h : Eq n 3
      hn : Membership.mem σ.cycleType n
      h1 : Eq (σ.cycleType.erase n) 0
      ⊢ σ.IsThreeCycle
    -/
    rw [IsThreeCycle, ← cons_erase hn, h1, h, ← cons_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h : Eq σ.support.card 3
    h0 : Not (Eq σ.cycleType 0)
    n : Nat
    hn : Membership.mem σ.cycleType n
    h1 : Not (Eq (σ.cycleType.erase n) 0)
    ⊢ σ.IsThreeCycle
  -/
  obtain ⟨m, hm⟩ := exists_mem_of_ne_zero h1
  /-
    case neg.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h : Eq σ.support.card 3
    h0 : Not (Eq σ.cycleType 0)
    n : Nat
    hn : Membership.mem σ.cycleType n
    h1 : Not (Eq (σ.cycleType.erase n) 0)
    m : Nat
    hm : Membership.mem (σ.cycleType.erase n) m
    ⊢ σ.IsThreeCycle
  -/
  rw [← sum_cycleType, ← cons_erase hn, ← cons_erase hm, Multiset.sum_cons, Multiset.sum_cons] at h
  /-
    case neg.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h0 : Not (Eq σ.cycleType 0)
    n : Nat
    hn : Membership.mem σ.cycleType n
    h1 : Not (Eq (σ.cycleType.erase n) 0)
    m : Nat
    h : Eq (HAdd.hAdd n (HAdd.hAdd m ((σ.cycleType.erase n).erase m).sum)) 3
    hm : Membership.mem (σ.cycleType.erase n) m
    ⊢ σ.IsThreeCycle
  -/
  have : ∀ {k}, 2 ≤ m → 2 ≤ n → n + (m + k) = 3 → False := by omega
  /-
    case neg.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h0 : Not (Eq σ.cycleType 0)
    n : Nat
    hn : Membership.mem σ.cycleType n
    h1 : Not (Eq (σ.cycleType.erase n) 0)
    m : Nat
    h : Eq (HAdd.hAdd n (HAdd.hAdd m ((σ.cycleType.erase n).erase m).sum)) 3
    hm : Membership.mem (σ.cycleType.erase n) m
    this : ∀ {k : Nat}, LE.le 2 m → LE.le 2 n → Eq (HAdd.hAdd n (HAdd.hAdd m k)) 3 …
    ⊢ σ.IsThreeCycle
  -/
  cases this (two_le_of_mem_cycleType (mem_of_mem_erase hm)) (two_le_of_mem_cycleType hn) h
  /-
    🎉 no goals
  -/


theorem isCycle (h : IsThreeCycle σ) : IsCycle σ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h : σ.IsThreeCycle
    ⊢ σ.IsCycle
  -/
  rw [← card_cycleType_eq_one, h.cycleType, card_singleton]
  /-
    🎉 no goals
  -/


theorem sign (h : IsThreeCycle σ) : sign σ = 1 := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h : σ.IsThreeCycle
    ⊢ Eq (Equiv.Perm.sign σ) 1
  -/
  rw [Equiv.Perm.sign_of_cycleType, h.cycleType]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    h : σ.IsThreeCycle
    ⊢ Eq (HPow.hPow (-1) (HAdd.hAdd (Singleton.singleton 3).sum (Singleton.singlet …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem inv {f : Perm α} (h : IsThreeCycle f) : IsThreeCycle f⁻¹ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    h : f.IsThreeCycle
    ⊢ (Inv.inv f).IsThreeCycle
  -/
  rwa [IsThreeCycle, cycleType_inv]
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_iff {f : Perm α} : IsThreeCycle f⁻¹ ↔ IsThreeCycle f :=
  ⟨by
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      ⊢ (Inv.inv f).IsThreeCycle → f.IsThreeCycle
    -/
    rw [← inv_inv f]
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      ⊢ (Inv.inv (Inv.inv (Inv.inv f))).IsThreeCycle → (Inv.inv (Inv.inv f)).IsThree …
    -/
    apply inv, inv⟩
    /-
      🎉 no goals
    -/


theorem orderOf {g : Perm α} (ht : IsThreeCycle g) : orderOf g = 3 := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    ht : g.IsThreeCycle
    ⊢ Eq (_root_.orderOf g) 3
  -/
  rw [← lcm_cycleType, ht.cycleType, Multiset.lcm_singleton, normalize_eq]
  /-
    🎉 no goals
  -/


theorem isThreeCycle_sq {g : Perm α} (ht : IsThreeCycle g) : IsThreeCycle (g * g) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    ht : g.IsThreeCycle
    ⊢ (HMul.hMul g g).IsThreeCycle
  -/
  rw [← pow_two, ← card_support_eq_three_iff, support_pow_coprime, ht.card_support]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    ht : g.IsThreeCycle
    ⊢ Nat.Coprime 2 (_root_.orderOf g)
  -/
  rw [ht.orderOf]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    ht : g.IsThreeCycle
    ⊢ Nat.Coprime 2 3
  -/
  norm_num
  /-
    🎉 no goals
  -/


theorem isThreeCycle_swap_mul_swap_same {a b c : α} (ab : a ≠ b) (ac : a ≠ c) (bc : b ≠ c) :
    IsThreeCycle (swap a b * swap a c) := by
  suffices h : support (swap a b * swap a c) = {a, b, c} by
    rw [← card_support_eq_three_iff, h]
    simp [ab, ac, bc]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    a b c : α
    ab : Ne a b
    ac : Ne a c
    bc : Ne b c
    ⊢ Eq (HMul.hMul (Equiv.swap a b) (Equiv.swap a c)).support (Insert.insert a (I …
  -/
  apply le_antisymm ((support_mul_le _ _).trans fun x => _) fun x hx => ?_
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      a b c : α
      ab : Ne a b
      ac : Ne a c
      bc : Ne b c
      ⊢ ∀ (x : α), Membership.mem (Max.max (Equiv.swap a b).support (Equiv.swap a c) …
    -/
  · simp [ab, ac, bc]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      a b c : α
      ab : Ne a b
      ac : Ne a c
      bc : Ne b c
      x : α
      hx : Membership.mem (Insert.insert a (Insert.insert b (Singleton.singleton c)) …
      ⊢ Membership.mem (HMul.hMul (Equiv.swap a b) (Equiv.swap a c)).support x
    -/
  · simp only [Finset.mem_insert, Finset.mem_singleton] at hx
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      a b c : α
      ab : Ne a b
      ac : Ne a c
      bc : Ne b c
      x : α
      hx : Or (Eq x a) (Or (Eq x b) (Eq x c))
      ⊢ Membership.mem (HMul.hMul (Equiv.swap a b) (Equiv.swap a c)).support x
    -/
    rw [mem_support]
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      a b c : α
      ab : Ne a b
      ac : Ne a c
      bc : Ne b c
      x : α
      hx : Or (Eq x a) (Or (Eq x b) (Eq x c))
      ⊢ Ne ((HMul.hMul (Equiv.swap a b) (Equiv.swap a c)) x) x
    -/
    simp only [Perm.coe_mul, Function.comp_apply, Ne]
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      a b c : α
      ab : Ne a b
      ac : Ne a c
      bc : Ne b c
      x : α
      hx : Or (Eq x a) (Or (Eq x b) (Eq x c))
      ⊢ Not (Eq ((Equiv.swap a b) ((Equiv.swap a c) x)) x)
    -/
    obtain rfl | rfl | rfl := hx
      /-
        case inl
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        b c : α
        bc : Ne b c
        x : α
        ab : Ne x b
        ac : Ne x c
        ⊢ Not (Eq ((Equiv.swap x b) ((Equiv.swap x c) x)) x)
      -/
    · rw [swap_apply_left, swap_apply_of_ne_of_ne ac.symm bc.symm]
      /-
        case inl
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        b c : α
        bc : Ne b c
        x : α
        ab : Ne x b
        ac : Ne x c
        ⊢ Not (Eq c x)
      -/
      exact ac.symm
      /-
        🎉 no goals
      -/
      /-
        case inr.inl
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        a c : α
        ac : Ne a c
        x : α
        ab : Ne a x
        bc : Ne x c
        ⊢ Not (Eq ((Equiv.swap a x) ((Equiv.swap a c) x)) x)
      -/
    · rw [swap_apply_of_ne_of_ne ab.symm bc, swap_apply_right]
      /-
        case inr.inl
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        a c : α
        ac : Ne a c
        x : α
        ab : Ne a x
        bc : Ne x c
        ⊢ Not (Eq a x)
      -/
      exact ab
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        a b : α
        ab : Ne a b
        x : α
        ac : Ne a x
        bc : Ne b x
        ⊢ Not (Eq ((Equiv.swap a b) ((Equiv.swap a x) x)) x)
      -/
    · rw [swap_apply_right, swap_apply_left]
      /-
        case inr.inr
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        a b : α
        ab : Ne a b
        x : α
        ac : Ne a x
        bc : Ne b x
        ⊢ Not (Eq b x)
      -/
      exact bc
      /-
        🎉 no goals
      -/


theorem swap_mul_swap_same_mem_closure_three_cycles {a b c : α} (ab : a ≠ b) (ac : a ≠ c) :
    swap a b * swap a c ∈ closure { σ : Perm α | IsThreeCycle σ } := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    a b c : α
    ab : Ne a b
    ac : Ne a c
    ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (HMul.hMul …
  -/
  by_cases bc : b = c
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      a b c : α
      ab : Ne a b
      ac : Ne a c
      bc : Eq b c
      ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (HMul.hMul …
    -/
  · subst bc
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      a b : α
      ab ac : Ne a b
      ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (HMul.hMul …
    -/
    simp [one_mem]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    a b c : α
    ab : Ne a b
    ac : Ne a c
    bc : Not (Eq b c)
    ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (HMul.hMul …
  -/
  exact subset_closure (isThreeCycle_swap_mul_swap_same ab ac bc)
  /-
    🎉 no goals
  -/


theorem IsSwap.mul_mem_closure_three_cycles {σ τ : Perm α} (hσ : IsSwap σ) (hτ : IsSwap τ) :
    σ * τ ∈ closure { σ : Perm α | IsThreeCycle σ } := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    hσ : σ.IsSwap
    hτ : τ.IsSwap
    ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (HMul.hMul …
  -/
  obtain ⟨a, b, ab, rfl⟩ := hσ
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    τ : Equiv.Perm α
    hτ : τ.IsSwap
    a b : α
    ab : Ne a b
    ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (HMul.hMul …
  -/
  obtain ⟨c, d, cd, rfl⟩ := hτ
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    a b : α
    ab : Ne a b
    c d : α
    cd : Ne c d
    ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (HMul.hMul …
  -/
  by_cases ac : a = c
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      a b : α
      ab : Ne a b
      c d : α
      cd : Ne c d
      ac : Eq a c
      ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (HMul.hMul …
    -/
  · subst ac
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      a b : α
      ab : Ne a b
      d : α
      cd : Ne a d
      ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (HMul.hMul …
    -/
    exact swap_mul_swap_same_mem_closure_three_cycles ab cd
    /-
      🎉 no goals
    -/
  have h' : swap a b * swap c d = swap a b * swap a c * (swap c a * swap c d) := by
    simp [swap_comm c a, mul_assoc]
  /-
    case neg
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    a b : α
    ab : Ne a b
    c d : α
    cd : Ne c d
    ac : Not (Eq a c)
    h' : Eq (HMul.hMul (Equiv.swap a b) (Equiv.swap c d)) (HMul.hMul (HMul.hMul (E …
    ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (HMul.hMul …
  -/
  rw [h']
  exact
    mul_mem (swap_mul_swap_same_mem_closure_three_cycles ab ac)
      (swap_mul_swap_same_mem_closure_three_cycles (Ne.symm ac) cd)


