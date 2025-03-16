/-- Non-dependent special case of `DFinsupp.addZeroClass` to help typeclass search. -/
instance addZeroClass' {β} [AddZeroClass β] : AddZeroClass (Π₀ _ : α, β) :=
  @DFinsupp.addZeroClass α (fun _ ↦ β) _


/-- A DFinsupp version of `Finsupp.toMultiset`. -/
def toMultiset : (Π₀ _ : α, ℕ) →+ Multiset α :=
  DFinsupp.sumAddHom fun a : α ↦ Multiset.replicateAddMonoidHom a


@[simp]
theorem toMultiset_single (a : α) (n : ℕ) :
    toMultiset (DFinsupp.single a n) = Multiset.replicate n a :=
  DFinsupp.sumAddHom_single _ _ _


/-- A DFinsupp version of `Multiset.toFinsupp`. -/
def toDFinsupp : Multiset α →+ Π₀ _ : α, ℕ where
  toFun s :=
    { toFun := fun n ↦ s.count n
      support' := Trunc.mk ⟨s, fun i ↦ (em (i ∈ s)).imp_right Multiset.count_eq_zero_of_not_mem⟩ }
  map_zero' := rfl
  map_add' _ _ := DFinsupp.ext fun _ ↦ Multiset.count_add _ _ _


@[simp]
theorem toDFinsupp_apply (s : Multiset α) (a : α) : Multiset.toDFinsupp s a = s.count a :=
  rfl


@[simp]
theorem toDFinsupp_support (s : Multiset α) : s.toDFinsupp.support = s.toFinset :=
  Finset.filter_true_of_mem fun _ hx ↦ count_ne_zero.mpr <| Multiset.mem_toFinset.1 hx


@[simp]
theorem toDFinsupp_replicate (a : α) (n : ℕ) :
    toDFinsupp (Multiset.replicate n a) = DFinsupp.single a n := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    n : Nat
    ⊢ Eq (Multiset.toDFinsupp (Multiset.replicate n a)) (DFinsupp.single a n)
  -/
  ext i
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    n : Nat
    i : α
    ⊢ Eq ((Multiset.toDFinsupp (Multiset.replicate n a)) i) ((DFinsupp.single a n) …
  -/
  dsimp [toDFinsupp]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    n : Nat
    i : α
    ⊢ Eq (Multiset.count i (Multiset.replicate n a)) ((DFinsupp.single a n) i)
  -/
  simp [count_replicate, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem toDFinsupp_singleton (a : α) : toDFinsupp {a} = DFinsupp.single a 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (Multiset.toDFinsupp (Singleton.singleton a)) (DFinsupp.single a 1)
  -/
  rw [← replicate_one, toDFinsupp_replicate]
  /-
    🎉 no goals
  -/


/-- `Multiset.toDFinsupp` as an `AddEquiv`. -/
@[simps! apply symm_apply]
def equivDFinsupp : Multiset α ≃+ Π₀ _ : α, ℕ :=
                                                                      /-
                                                                        α : Type u_1
                                                                        inst✝ : DecidableEq α
                                                                        s t : Multiset α
                                                                        ⊢ Eq (DFinsupp.toMultiset.comp Multiset.toDFinsupp) (AddMonoidHom.id (Multiset …
                                                                      -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  AddMonoidHom.toAddEquiv Multiset.toDFinsupp DFinsupp.toMultiset (by ext; simp) (by ext; simp)
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[simp]
theorem toDFinsupp_toMultiset (s : Multiset α) : DFinsupp.toMultiset (Multiset.toDFinsupp s) = s :=
  equivDFinsupp.symm_apply_apply s


theorem toDFinsupp_injective : Injective (toDFinsupp : Multiset α → Π₀ _a, ℕ) :=
  equivDFinsupp.injective


@[simp]
theorem toDFinsupp_inj : toDFinsupp s = toDFinsupp t ↔ s = t :=
  toDFinsupp_injective.eq_iff


@[simp]
theorem toDFinsupp_le_toDFinsupp : toDFinsupp s ≤ toDFinsupp t ↔ s ≤ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Iff (LE.le (Multiset.toDFinsupp s) (Multiset.toDFinsupp t)) (LE.le s t)
  -/
  simp [Multiset.le_iff_count, DFinsupp.le_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem toDFinsupp_lt_toDFinsupp : toDFinsupp s < toDFinsupp t ↔ s < t :=
  lt_iff_lt_of_le_iff_le' toDFinsupp_le_toDFinsupp toDFinsupp_le_toDFinsupp


@[simp]
theorem toDFinsupp_inter (s t : Multiset α) : toDFinsupp (s ∩ t) = toDFinsupp s ⊓ toDFinsupp t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (Multiset.toDFinsupp (Inter.inter s t)) (Min.min (Multiset.toDFinsupp s)  …
  -/
  ext i; simp
         /-
           🎉 no goals
         -/


@[simp]
theorem toDFinsupp_union (s t : Multiset α) : toDFinsupp (s ∪ t) = toDFinsupp s ⊔ toDFinsupp t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (Multiset.toDFinsupp (Union.union s t)) (Max.max (Multiset.toDFinsupp s)  …
  -/
  ext i; simp
         /-
           🎉 no goals
         -/


@[simp]
theorem toMultiset_toDFinsupp (f : Π₀ _ : α, ℕ) :
    Multiset.toDFinsupp (DFinsupp.toMultiset f) = f :=
  Multiset.equivDFinsupp.apply_symm_apply f


theorem toMultiset_injective : Injective (toMultiset : (Π₀ _a, ℕ) → Multiset α) :=
  Multiset.equivDFinsupp.symm.injective


@[simp]
theorem toMultiset_inj : toMultiset f = toMultiset g ↔ f = g :=
  toMultiset_injective.eq_iff


@[simp]
theorem toMultiset_le_toMultiset : toMultiset f ≤ toMultiset g ↔ f ≤ g := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    f g : DFinsupp fun _a => Nat
    ⊢ Iff (LE.le (DFinsupp.toMultiset f) (DFinsupp.toMultiset g)) (LE.le f g)
  -/
  simp_rw [← Multiset.toDFinsupp_le_toDFinsupp, toMultiset_toDFinsupp]
  /-
    🎉 no goals
  -/


@[simp]
theorem toMultiset_lt_toMultiset : toMultiset f < toMultiset g ↔ f < g := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    f g : DFinsupp fun _a => Nat
    ⊢ Iff (LT.lt (DFinsupp.toMultiset f) (DFinsupp.toMultiset g)) (LT.lt f g)
  -/
  simp_rw [← Multiset.toDFinsupp_lt_toDFinsupp, toMultiset_toDFinsupp]
  /-
    🎉 no goals
  -/


@[simp]
theorem toMultiset_inf : toMultiset (f ⊓ g) = toMultiset f ∩ toMultiset g :=
                                      /-
                                        α : Type u_1
                                        inst✝ : DecidableEq α
                                        f g : DFinsupp fun _a => Nat
                                        ⊢ Eq (Multiset.toDFinsupp (DFinsupp.toMultiset (Min.min f g))) (Multiset.toDFi …
                                      -/
  Multiset.toDFinsupp_injective <| by simp
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem toMultiset_sup : toMultiset (f ⊔ g) = toMultiset f∪ toMultiset g :=
                                      /-
                                        α : Type u_1
                                        inst✝ : DecidableEq α
                                        f g : DFinsupp fun _a => Nat
                                        ⊢ Eq (Multiset.toDFinsupp (DFinsupp.toMultiset (Max.max f g))) (Multiset.toDFi …
                                      -/
  Multiset.toDFinsupp_injective <| by simp
                                      /-
                                        🎉 no goals
                                      -/


