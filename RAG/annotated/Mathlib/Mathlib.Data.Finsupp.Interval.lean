/-- Pointwise `Singleton.singleton` bundled as a `Finsupp`. -/
@[simps]
def rangeSingleton (f : ι →₀ α) : ι →₀ Finset α where
  toFun i := {f i}
  support := f.support
  mem_support_toFun i := by
    /-
      ι : Type u_1
      α : Type u_2
      inst✝ : Zero α
      f✝ : Finsupp ι α
      i✝ : ι
      a : α
      f : Finsupp ι α
      i : ι
      ⊢ Iff (Membership.mem f.support i) (Ne ((fun i => Singleton.singleton (f i)) i …
    -/
    rw [← not_iff_not, not_mem_support_iff, not_ne_iff]
    /-
      ι : Type u_1
      α : Type u_2
      inst✝ : Zero α
      f✝ : Finsupp ι α
      i✝ : ι
      a : α
      f : Finsupp ι α
      i : ι
      ⊢ Iff (Eq (f i) 0) (Eq ((fun i => Singleton.singleton (f i)) i) 0)
    -/
    exact singleton_injective.eq_iff.symm
    /-
      🎉 no goals
    -/


theorem mem_rangeSingleton_apply_iff : a ∈ f.rangeSingleton i ↔ a = f i :=
  mem_singleton


open scoped Classical in
/-- Pointwise `Finset.Icc` bundled as a `Finsupp`. -/
@[simps toFun]
def rangeIcc (f g : ι →₀ α) : ι →₀ Finset α where
  toFun i := Icc (f i) (g i)
  support :=
    -- Porting note: Not needed (due to open scoped Classical), in mathlib3 too
    -- haveI := Classical.decEq ι
    f.support ∪ g.support
  mem_support_toFun i := by
    /-
      ι : Type u_1
      α : Type u_2
      inst✝² : Zero α
      inst✝¹ : PartialOrder α
      inst✝ : LocallyFiniteOrder α
      f✝ g✝ : Finsupp ι α
      i✝ : ι
      a : α
      f g : Finsupp ι α
      i : ι
      ⊢ Iff (Membership.mem (Union.union f.support g.support) i) (Ne ((fun i => Fins …
    -/
    rw [mem_union, ← not_iff_not, not_or, not_mem_support_iff, not_mem_support_iff, not_ne_iff]
    /-
      ι : Type u_1
      α : Type u_2
      inst✝² : Zero α
      inst✝¹ : PartialOrder α
      inst✝ : LocallyFiniteOrder α
      f✝ g✝ : Finsupp ι α
      i✝ : ι
      a : α
      f g : Finsupp ι α
      i : ι
      ⊢ Iff (And (Eq (f i) 0) (Eq (g i) 0)) (Eq ((fun i => Finset.Icc (f i) (g i)) i …
    -/
    exact Icc_eq_singleton_iff.symm
    /-
      🎉 no goals
    -/

-- Porting note: Added as alternative to rangeIcc_toFun to be used in proof of card_Icc

lemma coe_rangeIcc (f g : ι →₀ α) : rangeIcc f g i = Icc (f i) (g i) := rfl


open scoped Classical in
@[simp]
theorem rangeIcc_support (f g : ι →₀ α) :
    (rangeIcc f g).support = f.support ∪ g.support := rfl


theorem mem_rangeIcc_apply_iff : a ∈ f.rangeIcc g i ↔ f i ≤ a ∧ a ≤ g i := mem_Icc


open scoped Classical in
instance instLocallyFiniteOrder : LocallyFiniteOrder (ι →₀ α) :=
  -- Porting note: Not needed (due to open scoped Classical), in mathlib3 too
  -- haveI := Classical.decEq ι
  -- haveI := Classical.decEq α
  LocallyFiniteOrder.ofIcc (ι →₀ α) (fun f g => (f.support ∪ g.support).finsupp <| f.rangeIcc g)
    fun f g x => by
      refine
        (mem_finsupp_iff_of_support_subset <| Finset.subset_of_eq <| rangeIcc_support _ _).trans ?_
      /-
        ι : Type u_1
        α : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : Zero α
        inst✝ : LocallyFiniteOrder α
        f✝ g✝ f g x : Finsupp ι α
        ⊢ Iff (∀ (i : ι), Membership.mem ((f.rangeIcc g) i) (x i)) (And (LE.le f x) (L …
      -/
      simp_rw [mem_rangeIcc_apply_iff]
      /-
        ι : Type u_1
        α : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : Zero α
        inst✝ : LocallyFiniteOrder α
        f✝ g✝ f g x : Finsupp ι α
        ⊢ Iff (∀ (i : ι), And (LE.le (f i) (x i)) (LE.le (x i) (g i))) (And (LE.le f x …
      -/
      exact forall_and
      /-
        🎉 no goals
      -/


open scoped Classical in
theorem Icc_eq : Icc f g = (f.support ∪ g.support).finsupp (f.rangeIcc g) := rfl


open scoped Classical in
-- Porting note: removed [DecidableEq ι]
theorem card_Icc : #(Icc f g) = ∏ i ∈ f.support ∪ g.support, #(Icc (f i) (g i)):= by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : Zero α
    inst✝ : LocallyFiniteOrder α
    f g : Finsupp ι α
    ⊢ Eq (Finset.Icc f g).card ((Union.union f.support g.support).prod fun i => (F …
  -/
  simp_rw [Icc_eq, card_finsupp, coe_rangeIcc]
  /-
    🎉 no goals
  -/


open scoped Classical in
-- Porting note: removed [DecidableEq ι]
theorem card_Ico : #(Ico f g) = ∏ i ∈ f.support ∪ g.support, #(Icc (f i) (g i)) - 1 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : Zero α
    inst✝ : LocallyFiniteOrder α
    f g : Finsupp ι α
    ⊢ Eq (Finset.Ico f g).card (HSub.hSub ((Union.union f.support g.support).prod  …
  -/
  rw [card_Ico_eq_card_Icc_sub_one, card_Icc]
  /-
    🎉 no goals
  -/


open scoped Classical in
-- Porting note: removed [DecidableEq ι]
theorem card_Ioc : #(Ioc f g) = ∏ i ∈ f.support ∪ g.support, #(Icc (f i) (g i)) - 1 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : Zero α
    inst✝ : LocallyFiniteOrder α
    f g : Finsupp ι α
    ⊢ Eq (Finset.Ioc f g).card (HSub.hSub ((Union.union f.support g.support).prod  …
  -/
  rw [card_Ioc_eq_card_Icc_sub_one, card_Icc]
  /-
    🎉 no goals
  -/


open scoped Classical in
-- Porting note: removed [DecidableEq ι]
theorem card_Ioo : #(Ioo f g) = ∏ i ∈ f.support ∪ g.support, #(Icc (f i) (g i)) - 2 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : Zero α
    inst✝ : LocallyFiniteOrder α
    f g : Finsupp ι α
    ⊢ Eq (Finset.Ioo f g).card (HSub.hSub ((Union.union f.support g.support).prod  …
  -/
  rw [card_Ioo_eq_card_Icc_sub_two, card_Icc]
  /-
    🎉 no goals
  -/


open scoped Classical in
-- Porting note: removed [DecidableEq ι]
theorem card_uIcc :
    #(uIcc f g) = ∏ i ∈ f.support ∪ g.support, #(uIcc (f i) (g i)) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝² : Lattice α
    inst✝¹ : Zero α
    inst✝ : LocallyFiniteOrder α
    f g : Finsupp ι α
    ⊢ Eq (Finset.uIcc f g).card ((Union.union f.support g.support).prod fun i => ( …
  -/
  rw [← support_inf_union_support_sup]; exact card_Icc (_ : ι →₀ α) _
                                        /-
                                          🎉 no goals
                                        -/


theorem card_Iic : #(Iic f) = ∏ i ∈ f.support, #(Iic (f i)) := by
  classical simp_rw [Iic_eq_Icc, card_Icc, Finsupp.bot_eq_zero, support_zero, empty_union,
      zero_apply, bot_eq_zero]


theorem card_Iio : #(Iio f) = ∏ i ∈ f.support, #(Iic (f i)) - 1 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : CanonicallyOrderedAddCommMonoid α
    inst✝ : LocallyFiniteOrder α
    f : Finsupp ι α
    ⊢ Eq (Finset.Iio f).card (HSub.hSub (f.support.prod fun i => (Finset.Iic (f i) …
  -/
  rw [card_Iio_eq_card_Iic_sub_one, card_Iic]
  /-
    🎉 no goals
  -/


