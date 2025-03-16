instance instLocallyFiniteOrder : LocallyFiniteOrder (∀ i, α i) :=
  LocallyFiniteOrder.ofIcc _ (fun a b => piFinset fun i => Icc (a i) (b i)) fun a b x => by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝⁴ : Fintype ι
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → DecidableEq (α i)
      inst✝¹ : (i : ι) → PartialOrder (α i)
      inst✝ : (i : ι) → LocallyFiniteOrder (α i)
      a b x : (i : ι) → α i
      ⊢ Iff (Membership.mem ((fun a b => Fintype.piFinset fun i => Finset.Icc (a i)  …
    -/
    simp_rw [mem_piFinset, mem_Icc, le_def, forall_and]
    /-
      🎉 no goals
    -/


theorem Icc_eq : Icc a b = piFinset fun i => Icc (a i) (b i) :=
  rfl


theorem card_Icc : #(Icc a b) = ∏ i, #(Icc (a i) (b i)) :=
  card_piFinset _


theorem card_Ico : #(Ico a b) = ∏ i, #(Icc (a i) (b i)) - 1 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : Fintype ι
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → DecidableEq (α i)
    inst✝¹ : (i : ι) → PartialOrder (α i)
    inst✝ : (i : ι) → LocallyFiniteOrder (α i)
    a b : (i : ι) → α i
    ⊢ Eq (Finset.Ico a b).card (HSub.hSub (Finset.univ.prod fun i => (Finset.Icc ( …
  -/
  rw [card_Ico_eq_card_Icc_sub_one, card_Icc]
  /-
    🎉 no goals
  -/


theorem card_Ioc : #(Ioc a b) = ∏ i, #(Icc (a i) (b i)) - 1 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : Fintype ι
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → DecidableEq (α i)
    inst✝¹ : (i : ι) → PartialOrder (α i)
    inst✝ : (i : ι) → LocallyFiniteOrder (α i)
    a b : (i : ι) → α i
    ⊢ Eq (Finset.Ioc a b).card (HSub.hSub (Finset.univ.prod fun i => (Finset.Icc ( …
  -/
  rw [card_Ioc_eq_card_Icc_sub_one, card_Icc]
  /-
    🎉 no goals
  -/


theorem card_Ioo : #(Ioo a b) = ∏ i, #(Icc (a i) (b i)) - 2 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝⁴ : Fintype ι
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → DecidableEq (α i)
    inst✝¹ : (i : ι) → PartialOrder (α i)
    inst✝ : (i : ι) → LocallyFiniteOrder (α i)
    a b : (i : ι) → α i
    ⊢ Eq (Finset.Ioo a b).card (HSub.hSub (Finset.univ.prod fun i => (Finset.Icc ( …
  -/
  rw [card_Ioo_eq_card_Icc_sub_two, card_Icc]
  /-
    🎉 no goals
  -/


instance instLocallyFiniteOrderBot : LocallyFiniteOrderBot (∀ i, α i) :=
  .ofIic _ (fun b => piFinset fun i => Iic (b i)) fun b x => by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝⁴ : Fintype ι
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → DecidableEq (α i)
      inst✝¹ : (i : ι) → PartialOrder (α i)
      inst✝ : (i : ι) → LocallyFiniteOrderBot (α i)
      b✝ b x : (i : ι) → α i
      ⊢ Iff (Membership.mem ((fun b => Fintype.piFinset fun i => Finset.Iic (b i)) b …
    -/
    simp_rw [mem_piFinset, mem_Iic, le_def]
    /-
      🎉 no goals
    -/


lemma card_Iic : #(Iic b) = ∏ i, #(Iic (b i)) := card_piFinset _

                                                        /-
                                                          ι : Type u_1
                                                          α : ι → Type u_2
                                                          inst✝⁴ : Fintype ι
                                                          inst✝³ : DecidableEq ι
                                                          inst✝² : (i : ι) → DecidableEq (α i)
                                                          inst✝¹ : (i : ι) → PartialOrder (α i)
                                                          inst✝ : (i : ι) → LocallyFiniteOrderBot (α i)
                                                          b : (i : ι) → α i
                                                          ⊢ Eq (Finset.Iio b).card (HSub.hSub (Finset.univ.prod fun i => (Finset.Iic (b  …
                                                        -/
lemma card_Iio : #(Iio b) = ∏ i, #(Iic (b i)) - 1 := by rw [card_Iio_eq_card_Iic_sub_one, card_Iic]
                                                        /-
                                                          🎉 no goals
                                                        -/


instance instLocallyFiniteOrderTop : LocallyFiniteOrderTop (∀ i, α i) :=
  LocallyFiniteOrderTop.ofIci _ (fun a => piFinset fun i => Ici (a i)) fun a x => by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝⁴ : Fintype ι
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → DecidableEq (α i)
      inst✝¹ : (i : ι) → PartialOrder (α i)
      inst✝ : (i : ι) → LocallyFiniteOrderTop (α i)
      a✝ a x : (i : ι) → α i
      ⊢ Iff (Membership.mem ((fun a => Fintype.piFinset fun i => Finset.Ici (a i)) a …
    -/
    simp_rw [mem_piFinset, mem_Ici, le_def]
    /-
      🎉 no goals
    -/


lemma card_Ici : #(Ici a) = ∏ i, #(Ici (a i)) := card_piFinset _

                                                        /-
                                                          ι : Type u_1
                                                          α : ι → Type u_2
                                                          inst✝⁴ : Fintype ι
                                                          inst✝³ : DecidableEq ι
                                                          inst✝² : (i : ι) → DecidableEq (α i)
                                                          inst✝¹ : (i : ι) → PartialOrder (α i)
                                                          inst✝ : (i : ι) → LocallyFiniteOrderTop (α i)
                                                          a : (i : ι) → α i
                                                          ⊢ Eq (Finset.Ioi a).card (HSub.hSub (Finset.univ.prod fun i => (Finset.Ici (a  …
                                                        -/
lemma card_Ioi : #(Ioi a) = ∏ i, #(Ici (a i)) - 1 := by rw [card_Ioi_eq_card_Ici_sub_one, card_Ici]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem uIcc_eq : uIcc a b = piFinset fun i => uIcc (a i) (b i) := rfl


theorem card_uIcc : #(uIcc a b) = ∏ i, #(uIcc (a i) (b i)) := card_Icc _ _


