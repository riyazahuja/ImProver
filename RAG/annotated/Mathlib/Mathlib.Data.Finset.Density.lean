/-- Density of a finset.

`dens s` is the number of elements of `s` divided by the size of the ambient type `α`. -/
def dens (s : Finset α) : ℚ≥0 := s.card / Fintype.card α


lemma dens_eq_card_div_card (s : Finset α) : dens s = s.card / Fintype.card α := rfl


                                                         /-
                                                           α : Type u_2
                                                           inst✝ : Fintype α
                                                           ⊢ Eq EmptyCollection.emptyCollection.dens 0
                                                         -/
@[simp] lemma dens_empty : dens (∅ : Finset α) = 0 := by simp [dens]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp] lemma dens_singleton (a : α) : dens ({a} : Finset α) = (Fintype.card α : ℚ≥0)⁻¹ := by
  /-
    α : Type u_2
    inst✝ : Fintype α
    a : α
    ⊢ Eq (Singleton.singleton a).dens (Inv.inv ↑(Fintype.card α))
  -/
  simp [dens]
  /-
    🎉 no goals
  -/


@[simp] lemma dens_cons (h : a ∉ s) : (s.cons a h).dens = dens s + (Fintype.card α : ℚ≥0)⁻¹ := by
  /-
    α : Type u_2
    inst✝ : Fintype α
    s : Finset α
    a : α
    h : Not (Membership.mem s a)
    ⊢ Eq (Finset.cons a s h).dens (HAdd.hAdd s.dens (Inv.inv ↑(Fintype.card α)))
  -/
  simp [dens, add_div]
  /-
    🎉 no goals
  -/


@[simp] lemma dens_disjUnion (s t : Finset α) (h) : dens (s.disjUnion t h) = dens s + dens t := by
  /-
    α : Type u_2
    inst✝ : Fintype α
    s t : Finset α
    h : Disjoint s t
    ⊢ Eq (s.disjUnion t h).dens (HAdd.hAdd s.dens t.dens)
  -/
  simp_rw [dens, card_disjUnion, Nat.cast_add, add_div]
  /-
    🎉 no goals
  -/


@[simp] lemma dens_eq_zero : dens s = 0 ↔ s = ∅ := by
  /-
    α : Type u_2
    inst✝ : Fintype α
    s : Finset α
    ⊢ Iff (Eq s.dens 0) (Eq s EmptyCollection.emptyCollection)
  -/
  simp +contextual [dens, Fintype.card_eq_zero_iff, eq_empty_of_isEmpty]
  /-
    🎉 no goals
  -/


lemma dens_ne_zero : dens s ≠ 0 ↔ s.Nonempty := dens_eq_zero.not.trans nonempty_iff_ne_empty.symm


@[simp] lemma dens_pos : 0 < dens s ↔ s.Nonempty := pos_iff_ne_zero.trans dens_ne_zero


protected alias ⟨_, Nonempty.dens_pos⟩ := dens_pos

protected alias ⟨_, Nonempty.dens_ne_zero⟩ := dens_ne_zero


lemma dens_le_dens (h : s ⊆ t) : dens s ≤ dens t :=
                                                          /-
                                                            α : Type u_2
                                                            inst✝ : Fintype α
                                                            s t : Finset α
                                                            h : HasSubset.Subset s t
                                                            ⊢ LE.le 0 ↑(Fintype.card α)
                                                          -/
  div_le_div_of_nonneg_right (mod_cast card_mono h) <| by positivity
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma dens_lt_dens (h : s ⊂ t) : dens s < dens t :=
  div_lt_div_of_pos_right (mod_cast card_strictMono h) <| by
    /-
      α : Type u_2
      inst✝ : Fintype α
      s t : Finset α
      h : HasSSubset.SSubset s t
      ⊢ LT.lt 0 ↑(Fintype.card α)
    -/
    cases isEmpty_or_nonempty α
      /-
        case inl
        α : Type u_2
        inst✝ : Fintype α
        s t : Finset α
        h : HasSSubset.SSubset s t
        h✝ : IsEmpty α
        ⊢ LT.lt 0 ↑(Fintype.card α)
      -/
    · simp [Subsingleton.elim s t, ssubset_irrfl] at h
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_2
        inst✝ : Fintype α
        s t : Finset α
        h : HasSSubset.SSubset s t
        h✝ : Nonempty α
        ⊢ LT.lt 0 ↑(Fintype.card α)
      -/
    · exact mod_cast Fintype.card_pos
      /-
        🎉 no goals
      -/


@[mono] lemma dens_mono : Monotone (dens : Finset α → ℚ≥0) := fun _ _ ↦ dens_le_dens

@[mono] lemma dens_strictMono : StrictMono (dens : Finset α → ℚ≥0) := fun _ _ ↦ dens_lt_dens


lemma dens_map_le [Fintype β] (f : α ↪ β) : dens (s.map f) ≤ dens s := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : Fintype β
    f : Function.Embedding α β
    ⊢ LE.le (Finset.map f s).dens s.dens
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝¹ : Fintype α
      s : Finset α
      inst✝ : Fintype β
      f : Function.Embedding α β
      h✝ : IsEmpty α
      ⊢ LE.le (Finset.map f s).dens s.dens
    -/
  · simp [Subsingleton.elim s ∅]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    β : Type u_3
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : Fintype β
    f : Function.Embedding α β
    h✝ : Nonempty α
    ⊢ LE.le (Finset.map f s).dens s.dens
  -/
  simp_rw [dens, card_map]
  /-
    case inr
    α : Type u_2
    β : Type u_3
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : Fintype β
    f : Function.Embedding α β
    h✝ : Nonempty α
    ⊢ LE.le (HDiv.hDiv ↑s.card ↑(Fintype.card β)) (HDiv.hDiv ↑s.card ↑(Fintype.car …
  -/
  gcongr
    /-
      case inr.ha
      α : Type u_2
      β : Type u_3
      inst✝¹ : Fintype α
      s : Finset α
      inst✝ : Fintype β
      f : Function.Embedding α β
      h✝ : Nonempty α
      ⊢ LE.le 0 ↑s.card
    -/
  · positivity
    /-
      🎉 no goals
    -/
    /-
      case inr.hc
      α : Type u_2
      β : Type u_3
      inst✝¹ : Fintype α
      s : Finset α
      inst✝ : Fintype β
      f : Function.Embedding α β
      h✝ : Nonempty α
      ⊢ LT.lt 0 ↑(Fintype.card α)
    -/
  · exact mod_cast Fintype.card_pos
    /-
      🎉 no goals
    -/
    /-
      case inr.h.h
      α : Type u_2
      β : Type u_3
      inst✝¹ : Fintype α
      s : Finset α
      inst✝ : Fintype β
      f : Function.Embedding α β
      h✝ : Nonempty α
      ⊢ LE.le (Fintype.card α) (Fintype.card β)
    -/
  · exact Fintype.card_le_of_injective _ f.2
    /-
      🎉 no goals
    -/


@[simp] lemma dens_map_equiv [Fintype β] (e : α ≃ β) : (s.map e.toEmbedding).dens = s.dens := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : Fintype β
    e : Equiv α β
    ⊢ Eq (Finset.map e.toEmbedding s).dens s.dens
  -/
  simp [dens, Fintype.card_congr e]
  /-
    🎉 no goals
  -/


lemma dens_image [Fintype β] [DecidableEq β] {f : α → β} (hf : Bijective f) (s : Finset α) :
    (s.image f).dens = s.dens := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Fintype α
    inst✝¹ : Fintype β
    inst✝ : DecidableEq β
    f : α → β
    hf : Function.Bijective f
    s : Finset α
    ⊢ Eq (Finset.image f s).dens s.dens
  -/
  simpa [map_eq_image, -dens_map_equiv] using dens_map_equiv (.ofBijective f hf)
  /-
    🎉 no goals
  -/


@[simp] lemma card_mul_dens (s : Finset α) : Fintype.card α * s.dens = s.card := by
  /-
    α : Type u_2
    inst✝ : Fintype α
    s : Finset α
    ⊢ Eq (HMul.hMul (↑(Fintype.card α)) s.dens) ↑s.card
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_2
      inst✝ : Fintype α
      s : Finset α
      h✝ : IsEmpty α
      ⊢ Eq (HMul.hMul (↑(Fintype.card α)) s.dens) ↑s.card
    -/
  · simp [Subsingleton.elim s ∅]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    inst✝ : Fintype α
    s : Finset α
    h✝ : Nonempty α
    ⊢ Eq (HMul.hMul (↑(Fintype.card α)) s.dens) ↑s.card
  -/
  rw [dens, mul_div_cancel₀]
  /-
    case inr.hb
    α : Type u_2
    inst✝ : Fintype α
    s : Finset α
    h✝ : Nonempty α
    ⊢ Ne (↑(Fintype.card α)) 0
  -/
  exact mod_cast Fintype.card_ne_zero
  /-
    🎉 no goals
  -/


@[simp] lemma dens_mul_card (s : Finset α) : s.dens * Fintype.card α = s.card := by
  /-
    α : Type u_2
    inst✝ : Fintype α
    s : Finset α
    ⊢ Eq (HMul.hMul s.dens ↑(Fintype.card α)) ↑s.card
  -/
  rw [mul_comm, card_mul_dens]
  /-
    🎉 no goals
  -/


@[simp] lemma natCast_card_mul_nnratCast_dens (s : Finset α) :
    (Fintype.card α * s.dens : 𝕜) = s.card := mod_cast s.card_mul_dens


@[simp] lemma nnratCast_dens_mul_natCast_card (s : Finset α) :
    (s.dens * Fintype.card α : 𝕜) = s.card := mod_cast s.dens_mul_card


@[norm_cast] lemma nnratCast_dens (s : Finset α) : (s.dens : 𝕜) = s.card / Fintype.card α := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : Fintype α
    inst✝¹ : Semifield 𝕜
    inst✝ : CharZero 𝕜
    s : Finset α
    ⊢ Eq (↑s.dens) (HDiv.hDiv ↑s.card ↑(Fintype.card α))
  -/
  simp [dens]
  /-
    🎉 no goals
  -/


                                                           /-
                                                             α : Type u_2
                                                             inst✝¹ : Fintype α
                                                             inst✝ : Nonempty α
                                                             ⊢ Eq Finset.univ.dens 1
                                                           -/
@[simp] lemma dens_univ : dens (univ : Finset α) = 1 := by simp [dens, card_univ]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp] lemma dens_eq_one : dens s = 1 ↔ s = univ := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : Nonempty α
    ⊢ Iff (Eq s.dens 1) (Eq s Finset.univ)
  -/
  simp [dens, div_eq_one_iff_eq, card_eq_iff_eq_univ]
  /-
    🎉 no goals
  -/


lemma dens_ne_one : dens s ≠ 1 ↔ s ≠ univ := dens_eq_one.not


@[simp] lemma dens_le_one : s.dens ≤ 1 := by
  /-
    α : Type u_2
    inst✝ : Fintype α
    s : Finset α
    ⊢ LE.le s.dens 1
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_2
      inst✝ : Fintype α
      s : Finset α
      h✝ : IsEmpty α
      ⊢ LE.le s.dens 1
    -/
  · simp [Subsingleton.elim s ∅]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      inst✝ : Fintype α
      s : Finset α
      h✝ : Nonempty α
      ⊢ LE.le s.dens 1
    -/
  · simpa using dens_le_dens s.subset_univ
    /-
      🎉 no goals
    -/


lemma dens_union_add_dens_inter (s t : Finset α) :
    dens (s ∪ t) + dens (s ∩ t) = dens s + dens t := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (HAdd.hAdd (Union.union s t).dens (Inter.inter s t).dens) (HAdd.hAdd s.de …
  -/
  simp_rw [dens, ← add_div, ← Nat.cast_add, card_union_add_card_inter]
  /-
    🎉 no goals
  -/


lemma dens_inter_add_dens_union (s t : Finset α) :
                                                        /-
                                                          α : Type u_2
                                                          inst✝¹ : Fintype α
                                                          inst✝ : DecidableEq α
                                                          s t : Finset α
                                                          ⊢ Eq (HAdd.hAdd (Inter.inter s t).dens (Union.union s t).dens) (HAdd.hAdd s.de …
                                                        -/
    dens (s ∩ t) + dens (s ∪ t) = dens s + dens t := by rw [add_comm, dens_union_add_dens_inter]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp] lemma dens_union_of_disjoint (h : Disjoint s t) : dens (s ∪ t) = dens s + dens t := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    s t : Finset α
    inst✝ : DecidableEq α
    h : Disjoint s t
    ⊢ Eq (Union.union s t).dens (HAdd.hAdd s.dens t.dens)
  -/
  rw [← disjUnion_eq_union s t h, dens_disjUnion _ _ _]
  /-
    🎉 no goals
  -/


lemma dens_sdiff_add_dens_eq_dens (h : s ⊆ t) : dens (t \ s) + dens s = dens t := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    s t : Finset α
    inst✝ : DecidableEq α
    h : HasSubset.Subset s t
    ⊢ Eq (HAdd.hAdd (SDiff.sdiff t s).dens s.dens) t.dens
  -/
  simp [dens, ← card_sdiff_add_card_eq_card h, add_div]
  /-
    🎉 no goals
  -/


lemma dens_sdiff_add_dens (s t : Finset α) : dens (s \ t) + dens t = (s ∪ t).dens := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (HAdd.hAdd (SDiff.sdiff s t).dens t.dens) (Union.union s t).dens
  -/
  rw [← dens_union_of_disjoint sdiff_disjoint, sdiff_union_self_eq_union]
  /-
    🎉 no goals
  -/


lemma dens_sdiff_comm (h : card s = card t) : dens (s \ t) = dens (t \ s) :=
  add_left_injective (dens t) <| by
    /-
      α : Type u_2
      inst✝¹ : Fintype α
      s t : Finset α
      inst✝ : DecidableEq α
      h : Eq s.card t.card
      ⊢ Eq ((fun x => HAdd.hAdd x t.dens) (SDiff.sdiff s t).dens) ((fun x => HAdd.hA …
    -/
    simp_rw [dens_sdiff_add_dens, union_comm s, ← dens_sdiff_add_dens, dens, h]
    /-
      🎉 no goals
    -/


@[simp]
lemma dens_sdiff_add_dens_inter (s t : Finset α) : dens (s \ t) + dens (s ∩ t) = dens s := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (HAdd.hAdd (SDiff.sdiff s t).dens (Inter.inter s t).dens) s.dens
  -/
  rw [← dens_union_of_disjoint (disjoint_sdiff_inter _ _), sdiff_union_inter]
  /-
    🎉 no goals
  -/


@[simp]
lemma dens_inter_add_dens_sdiff (s t : Finset α) : dens (s ∩ t) + dens (s \ t) = dens s := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (HAdd.hAdd (Inter.inter s t).dens (SDiff.sdiff s t).dens) s.dens
  -/
  rw [add_comm, dens_sdiff_add_dens_inter]
  /-
    🎉 no goals
  -/


lemma dens_filter_add_dens_filter_not_eq_dens {α : Type*} [Fintype α] {s : Finset α}
    (p : α → Prop) [DecidablePred p] [∀ x, Decidable (¬p x)] :
    dens (s.filter p) + dens (s.filter fun a ↦ ¬ p a) = dens s := by
  classical
  rw [← dens_union_of_disjoint (disjoint_filter_filter_neg ..), filter_union_filter_neg_eq]


lemma dens_union_le (s t : Finset α) : dens (s ∪ t) ≤ dens s + dens t :=
  dens_union_add_dens_inter s t ▸ le_add_of_nonneg_right zero_le'


lemma dens_le_dens_sdiff_add_dens : dens s ≤ dens (s \ t) + dens t :=
  dens_sdiff_add_dens s _ ▸ dens_le_dens subset_union_left


lemma dens_sdiff (h : s ⊆ t) : dens (t \ s) = dens t - dens s :=
  eq_tsub_of_add_eq (dens_sdiff_add_dens_eq_dens h)


lemma le_dens_sdiff (s t : Finset α) : dens t - dens s ≤ dens (t \ s) :=
  tsub_le_iff_right.2 dens_le_dens_sdiff_add_dens


