instance instLocallyFiniteOrder : LocallyFiniteOrder (Σ i, α i) where
  finsetIcc := sigmaLift fun _ => Icc
  finsetIco := sigmaLift fun _ => Ico
  finsetIoc := sigmaLift fun _ => Ioc
  finsetIoo := sigmaLift fun _ => Ioo
  finset_mem_Icc := fun ⟨i, a⟩ ⟨j, b⟩ ⟨k, c⟩ => by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : (i : ι) → LocallyFiniteOrder (α i)
      x✝² x✝¹ x✝ : Sigma fun i => α i
      i : ι
      a : α i
      j : ι
      b : α j
      k : ι
      c : α k
      ⊢ Iff (Membership.mem (Finset.sigmaLift (fun x => Finset.Icc) ⟨i, a⟩ ⟨j, b⟩) ⟨ …
    -/
    simp_rw [mem_sigmaLift, le_def, mem_Icc, exists_and_left, ← exists_and_right, ← exists_prop]
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : (i : ι) → LocallyFiniteOrder (α i)
      x✝² x✝¹ x✝ : Sigma fun i => α i
      i : ι
      a : α i
      j : ι
      b : α j
      k : ι
      c : α k
      ⊢ Iff (Exists fun h => Exists fun _h => Exists fun x => LE.le c (Eq.rec b ⋯))  …
    -/
    exact exists₂_congr fun _ _ => by constructor <;> rintro ⟨⟨⟩, ht⟩ <;> exact ⟨rfl, ht⟩
    /-
      🎉 no goals
    -/
  finset_mem_Ico := fun ⟨i, a⟩ ⟨j, b⟩ ⟨k, c⟩ => by
    simp_rw [mem_sigmaLift, le_def, lt_def, mem_Ico, exists_and_left, ← exists_and_right, ←
      exists_prop]
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : (i : ι) → LocallyFiniteOrder (α i)
      x✝² x✝¹ x✝ : Sigma fun i => α i
      i : ι
      a : α i
      j : ι
      b : α j
      k : ι
      c : α k
      ⊢ Iff (Exists fun h => Exists fun _h => Exists fun x => LT.lt c (Eq.rec b ⋯))  …
    -/
    exact exists₂_congr fun _ _ => by constructor <;> rintro ⟨⟨⟩, ht⟩ <;> exact ⟨rfl, ht⟩
    /-
      🎉 no goals
    -/
  finset_mem_Ioc := fun ⟨i, a⟩ ⟨j, b⟩ ⟨k, c⟩ => by
    simp_rw [mem_sigmaLift, le_def, lt_def, mem_Ioc, exists_and_left, ← exists_and_right, ←
      exists_prop]
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : (i : ι) → LocallyFiniteOrder (α i)
      x✝² x✝¹ x✝ : Sigma fun i => α i
      i : ι
      a : α i
      j : ι
      b : α j
      k : ι
      c : α k
      ⊢ Iff (Exists fun h => Exists fun _h => Exists fun x => LE.le c (Eq.rec b ⋯))  …
    -/
    exact exists₂_congr fun _ _ => by constructor <;> rintro ⟨⟨⟩, ht⟩ <;> exact ⟨rfl, ht⟩
    /-
      🎉 no goals
    -/
  finset_mem_Ioo := fun ⟨i, a⟩ ⟨j, b⟩ ⟨k, c⟩ => by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : (i : ι) → LocallyFiniteOrder (α i)
      x✝² x✝¹ x✝ : Sigma fun i => α i
      i : ι
      a : α i
      j : ι
      b : α j
      k : ι
      c : α k
      ⊢ Iff (Membership.mem (Finset.sigmaLift (fun x => Finset.Ioo) ⟨i, a⟩ ⟨j, b⟩) ⟨ …
    -/
    simp_rw [mem_sigmaLift, lt_def, mem_Ioo, exists_and_left, ← exists_and_right, ← exists_prop]
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : (i : ι) → LocallyFiniteOrder (α i)
      x✝² x✝¹ x✝ : Sigma fun i => α i
      i : ι
      a : α i
      j : ι
      b : α j
      k : ι
      c : α k
      ⊢ Iff (Exists fun h => Exists fun _h => Exists fun x => LT.lt c (Eq.rec b ⋯))  …
    -/
    exact exists₂_congr fun _ _ => by constructor <;> rintro ⟨⟨⟩, ht⟩ <;> exact ⟨rfl, ht⟩
    /-
      🎉 no goals
    -/


theorem card_Icc : #(Icc a b) = if h : a.1 = b.1 then #(Icc (h.rec a.2) b.2) else 0 :=
  card_sigmaLift (fun _ => Icc) _ _


theorem card_Ico : #(Ico a b) = if h : a.1 = b.1 then #(Ico (h.rec a.2) b.2) else 0 :=
  card_sigmaLift (fun _ => Ico) _ _


theorem card_Ioc : #(Ioc a b) = if h : a.1 = b.1 then #(Ioc (h.rec a.2) b.2) else 0 :=
  card_sigmaLift (fun _ => Ioc) _ _


theorem card_Ioo : #(Ioo a b) = if h : a.1 = b.1 then #(Ioo (h.rec a.2) b.2) else 0 :=
  card_sigmaLift (fun _ => Ioo) _ _


@[simp]
theorem Icc_mk_mk : Icc (⟨i, a⟩ : Sigma α) ⟨i, b⟩ = (Icc a b).map (Embedding.sigmaMk i) :=
  dif_pos rfl


@[simp]
theorem Ico_mk_mk : Ico (⟨i, a⟩ : Sigma α) ⟨i, b⟩ = (Ico a b).map (Embedding.sigmaMk i) :=
  dif_pos rfl


@[simp]
theorem Ioc_mk_mk : Ioc (⟨i, a⟩ : Sigma α) ⟨i, b⟩ = (Ioc a b).map (Embedding.sigmaMk i) :=
  dif_pos rfl


@[simp]
theorem Ioo_mk_mk : Ioo (⟨i, a⟩ : Sigma α) ⟨i, b⟩ = (Ioo a b).map (Embedding.sigmaMk i) :=
  dif_pos rfl


