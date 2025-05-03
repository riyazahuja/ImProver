theorem map_valEmbedding_univ : (Finset.univ : Finset (Fin n)).map Fin.valEmbedding = Iio n := by
  /-
    n : Nat
    ⊢ Eq (Finset.map Fin.valEmbedding Finset.univ) (Finset.Iio n)
  -/
  ext
  /-
    case h
    n a✝ : Nat
    ⊢ Iff (Membership.mem (Finset.map Fin.valEmbedding Finset.univ) a✝) (Membershi …
  -/
  simp [orderIsoSubtype.symm.surjective.exists, OrderIso.symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioi_zero_eq_map : Ioi (0 : Fin n.succ) = univ.map (Fin.succEmb _) :=
                      /-
                        n : Nat
                        ⊢ Eq ↑(Finset.Ioi 0) ↑(Finset.map (Fin.succEmb n) Finset.univ)
                      -/
  coe_injective <| by ext; simp [pos_iff_ne_zero]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem Iio_last_eq_map : Iio (Fin.last n) = Finset.univ.map Fin.castSuccEmb :=
                      /-
                        n : Nat
                        ⊢ Eq ↑(Finset.Iio (Fin.last n)) ↑(Finset.map Fin.castSuccEmb Finset.univ)
                      -/
  coe_injective <| by ext; simp [lt_def]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem Ioi_succ (i : Fin n) : Ioi i.succ = (Ioi i).map (Fin.succEmb _) := by
  /-
    n : Nat
    i : Fin n
    ⊢ Eq (Finset.Ioi i.succ) (Finset.map (Fin.succEmb n) (Finset.Ioi i))
  -/
  ext i
  /-
    case h
    n : Nat
    i✝ : Fin n
    i : Fin (HAdd.hAdd n 1)
    ⊢ Iff (Membership.mem (Finset.Ioi i✝.succ) i) (Membership.mem (Finset.map (Fin …
  -/
  simp only [mem_filter, mem_Ioi, mem_map, mem_univ, Function.Embedding.coeFn_mk, exists_true_left]
  /-
    case h
    n : Nat
    i✝ : Fin n
    i : Fin (HAdd.hAdd n 1)
    ⊢ Iff (LT.lt i✝.succ i) (Exists fun a => And (LT.lt i✝ a) (Eq ((Fin.succEmb n) …
  -/
  constructor
    /-
      case h.mp
      n : Nat
      i✝ : Fin n
      i : Fin (HAdd.hAdd n 1)
      ⊢ LT.lt i✝.succ i → Exists fun a => And (LT.lt i✝ a) (Eq ((Fin.succEmb n) a) i)
    -/
  · refine cases ?_ ?_ i
      /-
        case h.mp.refine_1
        n : Nat
        i✝ : Fin n
        i : Fin (HAdd.hAdd n 1)
        ⊢ LT.lt i✝.succ 0 → Exists fun a => And (LT.lt i✝ a) (Eq ((Fin.succEmb n) a) 0)
      -/
    · rintro ⟨⟨⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mp.refine_2
        n : Nat
        i✝ : Fin n
        i : Fin (HAdd.hAdd n 1)
        ⊢ ∀ (i : Fin n), LT.lt i✝.succ i.succ → Exists fun a => And (LT.lt i✝ a) (Eq ( …
      -/
    · intro i hi
      /-
        case h.mp.refine_2
        n : Nat
        i✝¹ : Fin n
        i✝ : Fin (HAdd.hAdd n 1)
        i : Fin n
        hi : LT.lt i✝¹.succ i.succ
        ⊢ Exists fun a => And (LT.lt i✝¹ a) (Eq ((Fin.succEmb n) a) i.succ)
      -/
      exact ⟨i, succ_lt_succ_iff.mp hi, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      n : Nat
      i✝ : Fin n
      i : Fin (HAdd.hAdd n 1)
      ⊢ (Exists fun a => And (LT.lt i✝ a) (Eq ((Fin.succEmb n) a) i)) → LT.lt i✝.suc …
    -/
  · rintro ⟨i, hi, rfl⟩
    /-
      case h.mpr.intro.intro
      n : Nat
      i✝ i : Fin n
      hi : LT.lt i✝ i
      ⊢ LT.lt i✝.succ ((Fin.succEmb n) i)
    -/
    simpa
    /-
      🎉 no goals
    -/


@[simp]
theorem Iio_castSucc (i : Fin n) : Iio (castSucc i) = (Iio i).map Fin.castSuccEmb := by
  /-
    n : Nat
    i : Fin n
    ⊢ Eq (Finset.Iio i.castSucc) (Finset.map Fin.castSuccEmb (Finset.Iio i))
  -/
  apply Finset.map_injective Fin.valEmbedding
  /-
    case a
    n : Nat
    i : Fin n
    ⊢ Eq (Finset.map Fin.valEmbedding (Finset.Iio i.castSucc)) (Finset.map Fin.val …
  -/
  rw [Finset.map_map, Fin.map_valEmbedding_Iio]
  /-
    case a
    n : Nat
    i : Fin n
    ⊢ Eq (Finset.Iio ↑i.castSucc) (Finset.map (Fin.castSuccEmb.trans Fin.valEmbedd …
  -/
  exact (Fin.map_valEmbedding_Iio i).symm
  /-
    🎉 no goals
  -/


theorem card_filter_univ_succ (p : Fin (n + 1) → Prop) [DecidablePred p] :
    #{x | p x} = if p 0 then #{x | p (.succ x)} + 1 else #{x | p (.succ x)} := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1) → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Finset.filter (fun x => p x) Finset.univ).card (ite (p 0) (HAdd.hAdd (Fi …
  -/
  rw [Fin.univ_succ, filter_cons, apply_ite Finset.card, card_cons, filter_map, card_map]; rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


theorem card_filter_univ_succ' (p : Fin (n + 1) → Prop) [DecidablePred p] :
    #{x | p x} = ite (p 0) 1 0 + #{x | p (.succ x)}:= by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1) → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Finset.filter (fun x => p x) Finset.univ).card (HAdd.hAdd (ite (p 0) 1 0 …
  -/
                                            /-
                                              🎉 no goals
                                            -/
  rw [card_filter_univ_succ]; split_ifs <;> simp [add_comm]
                                            /-
                                              🎉 no goals
                                            -/


theorem card_filter_univ_eq_vector_get_eq_count [DecidableEq α] (a : α) (v : List.Vector α n) :
    #{i | v.get i = a} = v.toList.count a := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : DecidableEq α
    a : α
    v : List.Vector α n
    ⊢ Eq (Finset.filter (fun i => Eq (v.get i) a) Finset.univ).card (List.count a  …
  -/
  induction' v with n x xs hxs
    /-
      case nil
      α : Type u_1
      n : Nat
      inst✝ : DecidableEq α
      a : α
      ⊢ Eq (Finset.filter (fun i => Eq (List.Vector.nil.get i) a) Finset.univ).card  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp_rw [card_filter_univ_succ', Vector.get_cons_zero, Vector.toList_cons, Vector.get_cons_succ,
      hxs, List.count_cons, add_comm (ite (x = a) 1 0), beq_iff_eq]


