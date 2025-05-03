/-- UV-compression is injective on the elements it moves. See `UV.compress`. -/
theorem sup_sdiff_injOn [GeneralizedBooleanAlgebra α] (u v : α) :
    { x | Disjoint u x ∧ v ≤ x }.InjOn fun x => (x ⊔ u) \ v := by
  /-
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    u v : α
    ⊢ Set.InjOn (fun x => SDiff.sdiff (Max.max x u) v) (setOf fun x => And (Disjoi …
  -/
  rintro a ha b hb hab
  have h : ((a ⊔ u) \ v) \ u ⊔ v = ((b ⊔ u) \ v) \ u ⊔ v := by
    dsimp at hab
    rw [hab]
  rwa [sdiff_sdiff_comm, ha.1.symm.sup_sdiff_cancel_right, sdiff_sdiff_comm,
    hb.1.symm.sup_sdiff_cancel_right, sdiff_sup_cancel ha.2, sdiff_sup_cancel hb.2] at h

-- The namespace is here to distinguish from other compressions.

/-- UV-compressing `a` means removing `v` from it and adding `u` if `a` and `u` are disjoint and
`v ≤ a` (it replaces the `v` part of `a` by the `u` part). Else, UV-compressing `a` doesn't do
anything. This is most useful when `u` and `v` are disjoint finsets of the same size. -/
def compress (u v a : α) : α :=
  if Disjoint u a ∧ v ≤ a then (a ⊔ u) \ v else a


theorem compress_of_disjoint_of_le (hua : Disjoint u a) (hva : v ≤ a) :
    compress u v a = (a ⊔ u) \ v :=
  if_pos ⟨hua, hva⟩


theorem compress_of_disjoint_of_le' (hva : Disjoint v a) (hua : u ≤ a) :
    compress u v ((a ⊔ v) \ u) = a := by
  rw [compress_of_disjoint_of_le disjoint_sdiff_self_right
      (le_sdiff.2 ⟨(le_sup_right : v ≤ a ⊔ v), hva.mono_right hua⟩),
    sdiff_sup_cancel (le_sup_of_le_left hua), hva.symm.sup_sdiff_cancel_right]


@[simp]
theorem compress_self (u a : α) : compress u u a = a := by
  /-
    α : Type u_1
    inst✝² : GeneralizedBooleanAlgebra α
    inst✝¹ : DecidableRel Disjoint
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    u a : α
    ⊢ Eq (UV.compress u u a) a
  -/
  unfold compress
  /-
    α : Type u_1
    inst✝² : GeneralizedBooleanAlgebra α
    inst✝¹ : DecidableRel Disjoint
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    u a : α
    ⊢ Eq (ite (And (Disjoint u a) (LE.le u a)) (SDiff.sdiff (Max.max a u) u) a) a
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      inst✝² : GeneralizedBooleanAlgebra α
      inst✝¹ : DecidableRel Disjoint
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      u a : α
      h : And (Disjoint u a) (LE.le u a)
      ⊢ Eq (SDiff.sdiff (Max.max a u) u) a
    -/
  · exact h.1.symm.sup_sdiff_cancel_right
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : GeneralizedBooleanAlgebra α
      inst✝¹ : DecidableRel Disjoint
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      u a : α
      h : Not (And (Disjoint u a) (LE.le u a))
      ⊢ Eq a a
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- An element can be compressed to any other element by removing/adding the differences. -/
@[simp]
theorem compress_sdiff_sdiff (a b : α) : compress (a \ b) (b \ a) b = a := by
  /-
    α : Type u_1
    inst✝² : GeneralizedBooleanAlgebra α
    inst✝¹ : DecidableRel Disjoint
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b : α
    ⊢ Eq (UV.compress (SDiff.sdiff a b) (SDiff.sdiff b a) b) a
  -/
  refine (compress_of_disjoint_of_le disjoint_sdiff_self_left sdiff_le).trans ?_
  /-
    α : Type u_1
    inst✝² : GeneralizedBooleanAlgebra α
    inst✝¹ : DecidableRel Disjoint
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b : α
    ⊢ Eq (SDiff.sdiff (Max.max b (SDiff.sdiff a b)) (SDiff.sdiff b a)) a
  -/
  rw [sup_sdiff_self_right, sup_sdiff, disjoint_sdiff_self_right.sdiff_eq_left, sup_eq_right]
  /-
    α : Type u_1
    inst✝² : GeneralizedBooleanAlgebra α
    inst✝¹ : DecidableRel Disjoint
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b : α
    ⊢ LE.le (SDiff.sdiff b (SDiff.sdiff b a)) a
  -/
  exact sdiff_sdiff_le
  /-
    🎉 no goals
  -/


/-- Compressing an element is idempotent. -/
@[simp]
theorem compress_idem (u v a : α) : compress u v (compress u v a) = compress u v a := by
  /-
    α : Type u_1
    inst✝² : GeneralizedBooleanAlgebra α
    inst✝¹ : DecidableRel Disjoint
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    u v a : α
    ⊢ Eq (UV.compress u v (UV.compress u v a)) (UV.compress u v a)
  -/
  unfold compress
  /-
    α : Type u_1
    inst✝² : GeneralizedBooleanAlgebra α
    inst✝¹ : DecidableRel Disjoint
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    u v a : α
    ⊢ Eq (ite (And (Disjoint u (ite (And (Disjoint u a) (LE.le v a)) (SDiff.sdiff  …
  -/
  split_ifs with h h'
    /-
      case pos
      α : Type u_1
      inst✝² : GeneralizedBooleanAlgebra α
      inst✝¹ : DecidableRel Disjoint
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      u v a : α
      h : And (Disjoint u a) (LE.le v a)
      h' : And (Disjoint u (SDiff.sdiff (Max.max a u) v)) (LE.le v (SDiff.sdiff (Max …
      ⊢ Eq (SDiff.sdiff (Max.max (SDiff.sdiff (Max.max a u) v) u) v) (SDiff.sdiff (M …
    -/
  · rw [le_sdiff_iff.1 h'.2, sdiff_bot, sdiff_bot, sup_assoc, sup_idem]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : GeneralizedBooleanAlgebra α
      inst✝¹ : DecidableRel Disjoint
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      u v a : α
      h : And (Disjoint u a) (LE.le v a)
      h' : Not (And (Disjoint u (SDiff.sdiff (Max.max a u) v)) (LE.le v (SDiff.sdiff …
      ⊢ Eq (SDiff.sdiff (Max.max a u) v) (SDiff.sdiff (Max.max a u) v)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : GeneralizedBooleanAlgebra α
      inst✝¹ : DecidableRel Disjoint
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      u v a : α
      h : Not (And (Disjoint u a) (LE.le v a))
      ⊢ Eq a a
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- To UV-compress a set family, we compress each of its elements, except that we don't want to
reduce the cardinality, so we keep all elements whose compression is already present. -/
def compression (u v : α) (s : Finset α) :=
  {a ∈ s | compress u v a ∈ s} ∪ {a ∈ s.image <| compress u v | a ∉ s}


@[inherit_doc]
scoped[FinsetFamily] notation "𝓒 " => UV.compression


/-- `IsCompressed u v s` expresses that `s` is UV-compressed. -/
def IsCompressed (u v : α) (s : Finset α) :=
  𝓒 u v s = s


/-- UV-compression is injective on the sets that are not UV-compressed. -/
theorem compress_injOn : Set.InjOn (compress u v) ↑{a ∈ s | compress u v a ∉ s} := by
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v : α
    inst✝ : DecidableEq α
    ⊢ Set.InjOn (UV.compress u v) ↑(Finset.filter (fun a => Not (Membership.mem s  …
  -/
  intro a ha b hb hab
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v : α
    inst✝ : DecidableEq α
    a : α
    ha : Membership.mem (↑(Finset.filter (fun a => Not (Membership.mem s (UV.compr …
    b : α
    hb : Membership.mem (↑(Finset.filter (fun a => Not (Membership.mem s (UV.compr …
    hab : Eq (UV.compress u v a) (UV.compress u v b)
    ⊢ Eq a b
  -/
  rw [mem_coe, mem_filter] at ha hb
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v : α
    inst✝ : DecidableEq α
    a : α
    ha : And (Membership.mem s a) (Not (Membership.mem s (UV.compress u v a)))
    b : α
    hb : And (Membership.mem s b) (Not (Membership.mem s (UV.compress u v b)))
    hab : Eq (UV.compress u v a) (UV.compress u v b)
    ⊢ Eq a b
  -/
  rw [compress] at ha hab
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v : α
    inst✝ : DecidableEq α
    a : α
    ha : And (Membership.mem s a) (Not (Membership.mem s (ite (And (Disjoint u a)  …
    b : α
    hb : And (Membership.mem s b) (Not (Membership.mem s (UV.compress u v b)))
    hab : Eq (ite (And (Disjoint u a) (LE.le v a)) (SDiff.sdiff (Max.max a u) v) a …
    ⊢ Eq a b
  -/
  split_ifs at ha hab with has
    /-
      case pos
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v : α
      inst✝ : DecidableEq α
      a b : α
      hb : And (Membership.mem s b) (Not (Membership.mem s (UV.compress u v b)))
      has : And (Disjoint u a) (LE.le v a)
      ha : And (Membership.mem s a) (Not (Membership.mem s (SDiff.sdiff (Max.max a u …
      hab : Eq (SDiff.sdiff (Max.max a u) v) (UV.compress u v b)
      ⊢ Eq a b
    -/
  · rw [compress] at hb hab
    /-
      case pos
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v : α
      inst✝ : DecidableEq α
      a b : α
      hb : And (Membership.mem s b) (Not (Membership.mem s (ite (And (Disjoint u b)  …
      has : And (Disjoint u a) (LE.le v a)
      ha : And (Membership.mem s a) (Not (Membership.mem s (SDiff.sdiff (Max.max a u …
      hab : Eq (SDiff.sdiff (Max.max a u) v) (ite (And (Disjoint u b) (LE.le v b)) ( …
      ⊢ Eq a b
    -/
    split_ifs at hb hab with hbs
      /-
        case pos
        α : Type u_1
        inst✝³ : GeneralizedBooleanAlgebra α
        inst✝² : DecidableRel Disjoint
        inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
        s : Finset α
        u v : α
        inst✝ : DecidableEq α
        a b : α
        has : And (Disjoint u a) (LE.le v a)
        ha : And (Membership.mem s a) (Not (Membership.mem s (SDiff.sdiff (Max.max a u …
        hbs : And (Disjoint u b) (LE.le v b)
        hb : And (Membership.mem s b) (Not (Membership.mem s (SDiff.sdiff (Max.max b u …
        hab : Eq (SDiff.sdiff (Max.max a u) v) (SDiff.sdiff (Max.max b u) v)
        ⊢ Eq a b
      -/
    · exact sup_sdiff_injOn u v has hbs hab
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝³ : GeneralizedBooleanAlgebra α
        inst✝² : DecidableRel Disjoint
        inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
        s : Finset α
        u v : α
        inst✝ : DecidableEq α
        a b : α
        has : And (Disjoint u a) (LE.le v a)
        ha : And (Membership.mem s a) (Not (Membership.mem s (SDiff.sdiff (Max.max a u …
        hbs : Not (And (Disjoint u b) (LE.le v b))
        hb : And (Membership.mem s b) (Not (Membership.mem s b))
        hab : Eq (SDiff.sdiff (Max.max a u) v) b
        ⊢ Eq a b
      -/
    · exact (hb.2 hb.1).elim
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v : α
      inst✝ : DecidableEq α
      a b : α
      hb : And (Membership.mem s b) (Not (Membership.mem s (UV.compress u v b)))
      has : Not (And (Disjoint u a) (LE.le v a))
      ha : And (Membership.mem s a) (Not (Membership.mem s a))
      hab : Eq a (UV.compress u v b)
      ⊢ Eq a b
    -/
  · exact (ha.2 ha.1).elim
    /-
      🎉 no goals
    -/


/-- `a` is in the UV-compressed family iff it's in the original and its compression is in the
original, or it's not in the original but it's the compression of something in the original. -/
theorem mem_compression :
    a ∈ 𝓒 u v s ↔ a ∈ s ∧ compress u v a ∈ s ∨ a ∉ s ∧ ∃ b ∈ s, compress u v b = a := by
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ⊢ Iff (Membership.mem (UV.compression u v s) a) (Or (And (Membership.mem s a)  …
  -/
  simp_rw [compression, mem_union, mem_filter, mem_image, and_comm]
  /-
    🎉 no goals
  -/


protected theorem IsCompressed.eq (h : IsCompressed u v s) : 𝓒 u v s = s := h


@[simp]
theorem compression_self (u : α) (s : Finset α) : 𝓒 u u s = s := by
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableEq α
    u : α
    s : Finset α
    ⊢ Eq (UV.compression u u s) s
  -/
  unfold compression
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableEq α
    u : α
    s : Finset α
    ⊢ Eq (Union.union (Finset.filter (fun a => Membership.mem s (UV.compress u u a …
  -/
  convert union_empty s
    /-
      case h.e'_2.h.e'_3
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : DecidableEq α
      u : α
      s : Finset α
      ⊢ Eq (Finset.filter (fun a => Membership.mem s (UV.compress u u a)) s) s
    -/
  · ext a
    /-
      case h.e'_2.h.e'_3.h
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : DecidableEq α
      u : α
      s : Finset α
      a : α
      ⊢ Iff (Membership.mem (Finset.filter (fun a => Membership.mem s (UV.compress u …
    -/
    rw [mem_filter, compress_self, and_self_iff]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_4
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : DecidableEq α
      u : α
      s : Finset α
      ⊢ Eq (Finset.filter (fun a => Not (Membership.mem s a)) (Finset.image (UV.comp …
    -/
  · refine eq_empty_of_forall_not_mem fun a ha ↦ ?_
    /-
      case h.e'_2.h.e'_4
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : DecidableEq α
      u : α
      s : Finset α
      a : α
      ha : Membership.mem (Finset.filter (fun a => Not (Membership.mem s a)) (Finset …
      ⊢ False
    -/
    simp_rw [mem_filter, mem_image, compress_self] at ha
    /-
      case h.e'_2.h.e'_4
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : DecidableEq α
      u : α
      s : Finset α
      a : α
      ha : And (Exists fun a_1 => And (Membership.mem s a_1) (Eq a_1 a)) (Not (Membe …
      ⊢ False
    -/
    obtain ⟨⟨b, hb, rfl⟩, hb'⟩ := ha
    /-
      case h.e'_2.h.e'_4.intro.intro.intro
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      inst✝ : DecidableEq α
      u : α
      s : Finset α
      b : α
      hb : Membership.mem s b
      hb' : Not (Membership.mem s b)
      ⊢ False
    -/
    exact hb' hb
    /-
      🎉 no goals
    -/


/-- Any family is compressed along two identical elements. -/
theorem isCompressed_self (u : α) (s : Finset α) : IsCompressed u u s := compression_self u s


theorem compress_disjoint :
    Disjoint {a ∈ s | compress u v a ∈ s} {a ∈ s.image <| compress u v | a ∉ s} :=
  disjoint_left.2 fun _a ha₁ ha₂ ↦ (mem_filter.1 ha₂).2 (mem_filter.1 ha₁).1


theorem compress_mem_compression (ha : a ∈ s) : compress u v a ∈ 𝓒 u v s := by
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Membership.mem s a
    ⊢ Membership.mem (UV.compression u v s) (UV.compress u v a)
  -/
  rw [mem_compression]
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Membership.mem s a
    ⊢ Or (And (Membership.mem s (UV.compress u v a)) (Membership.mem s (UV.compres …
  -/
  by_cases h : compress u v a ∈ s
    /-
      case pos
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Membership.mem s a
      h : Membership.mem s (UV.compress u v a)
      ⊢ Or (And (Membership.mem s (UV.compress u v a)) (Membership.mem s (UV.compres …
    -/
  · rw [compress_idem]
    /-
      case pos
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Membership.mem s a
      h : Membership.mem s (UV.compress u v a)
      ⊢ Or (And (Membership.mem s (UV.compress u v a)) (Membership.mem s (UV.compres …
    -/
    exact Or.inl ⟨h, h⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Membership.mem s a
      h : Not (Membership.mem s (UV.compress u v a))
      ⊢ Or (And (Membership.mem s (UV.compress u v a)) (Membership.mem s (UV.compres …
    -/
  · exact Or.inr ⟨h, a, ha, rfl⟩
    /-
      🎉 no goals
    -/

-- This is a special case of `compress_mem_compression` once we have `compression_idem`.

theorem compress_mem_compression_of_mem_compression (ha : a ∈ 𝓒 u v s) :
    compress u v a ∈ 𝓒 u v s := by
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Membership.mem (UV.compression u v s) a
    ⊢ Membership.mem (UV.compression u v s) (UV.compress u v a)
  -/
  rw [mem_compression] at ha ⊢
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Or (And (Membership.mem s a) (Membership.mem s (UV.compress u v a))) (And …
    ⊢ Or (And (Membership.mem s (UV.compress u v a)) (Membership.mem s (UV.compres …
  -/
  simp only [compress_idem, exists_prop]
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Or (And (Membership.mem s a) (Membership.mem s (UV.compress u v a))) (And …
    ⊢ Or (And (Membership.mem s (UV.compress u v a)) (Membership.mem s (UV.compres …
  -/
  obtain ⟨_, ha⟩ | ⟨_, b, hb, rfl⟩ := ha
    /-
      case inl.intro
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      left✝ : Membership.mem s a
      ha : Membership.mem s (UV.compress u v a)
      ⊢ Or (And (Membership.mem s (UV.compress u v a)) (Membership.mem s (UV.compres …
    -/
  · exact Or.inl ⟨ha, ha⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.intro
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v : α
      inst✝ : DecidableEq α
      b : α
      hb : Membership.mem s b
      left✝ : Not (Membership.mem s (UV.compress u v b))
      ⊢ Or (And (Membership.mem s (UV.compress u v (UV.compress u v b))) (Membership …
    -/
  · exact Or.inr ⟨by rwa [compress_idem], b, hb, (compress_idem _ _ _).symm⟩
    /-
      🎉 no goals
    -/


/-- Compressing a family is idempotent. -/
@[simp]
theorem compression_idem (u v : α) (s : Finset α) : 𝓒 u v (𝓒 u v s) = 𝓒 u v s := by
  have h : {a ∈ 𝓒 u v s | compress u v a ∉ 𝓒 u v s} = ∅ :=
    filter_false_of_mem fun a ha h ↦ h <| compress_mem_compression_of_mem_compression ha
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableEq α
    u v : α
    s : Finset α
    h : Eq (Finset.filter (fun a => Not (Membership.mem (UV.compression u v s) (UV …
    ⊢ Eq (UV.compression u v (UV.compression u v s)) (UV.compression u v s)
  -/
  rw [compression, filter_image, h, image_empty, ← h]
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableEq α
    u v : α
    s : Finset α
    h : Eq (Finset.filter (fun a => Not (Membership.mem (UV.compression u v s) (UV …
    ⊢ Eq (Union.union (Finset.filter (fun a => Membership.mem (UV.compression u v  …
  -/
  exact filter_union_filter_neg_eq _ (compression u v s)
  /-
    🎉 no goals
  -/


/-- Compressing a family doesn't change its size. -/
@[simp]
theorem card_compression (u v : α) (s : Finset α) : #(𝓒 u v s) = #s := by
  rw [compression, card_union_of_disjoint compress_disjoint, filter_image,
    card_image_of_injOn compress_injOn, ← card_union_of_disjoint (disjoint_filter_filter_neg s _ _),
    filter_union_filter_neg_eq]


theorem le_of_mem_compression_of_not_mem (h : a ∈ 𝓒 u v s) (ha : a ∉ s) : u ≤ a := by
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    h : Membership.mem (UV.compression u v s) a
    ha : Not (Membership.mem s a)
    ⊢ LE.le u a
  -/
  rw [mem_compression] at h
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    h : Or (And (Membership.mem s a) (Membership.mem s (UV.compress u v a))) (And  …
    ha : Not (Membership.mem s a)
    ⊢ LE.le u a
  -/
  obtain h | ⟨-, b, hb, hba⟩ := h
    /-
      case inl
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      h : And (Membership.mem s a) (Membership.mem s (UV.compress u v a))
      ⊢ LE.le u a
    -/
  · cases ha h.1
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.intro.intro
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Not (Membership.mem s a)
    b : α
    hb : Membership.mem s b
    hba : Eq (UV.compress u v b) a
    ⊢ LE.le u a
  -/
  unfold compress at hba
  /-
    case inr.intro.intro.intro
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Not (Membership.mem s a)
    b : α
    hb : Membership.mem s b
    hba : Eq (ite (And (Disjoint u b) (LE.le v b)) (SDiff.sdiff (Max.max b u) v) b …
    ⊢ LE.le u a
  -/
  split_ifs at hba with h
    /-
      case pos
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      b : α
      hb : Membership.mem s b
      h : And (Disjoint u b) (LE.le v b)
      hba : Eq (SDiff.sdiff (Max.max b u) v) a
      ⊢ LE.le u a
    -/
  · rw [← hba, le_sdiff]
    /-
      case pos
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      b : α
      hb : Membership.mem s b
      h : And (Disjoint u b) (LE.le v b)
      hba : Eq (SDiff.sdiff (Max.max b u) v) a
      ⊢ And (LE.le u (Max.max b u)) (Disjoint u v)
    -/
    exact ⟨le_sup_right, h.1.mono_right h.2⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      b : α
      hb : Membership.mem s b
      h : Not (And (Disjoint u b) (LE.le v b))
      hba : Eq b a
      ⊢ LE.le u a
    -/
  · cases ne_of_mem_of_not_mem hb ha hba
    /-
      🎉 no goals
    -/


theorem disjoint_of_mem_compression_of_not_mem (h : a ∈ 𝓒 u v s) (ha : a ∉ s) : Disjoint v a := by
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    h : Membership.mem (UV.compression u v s) a
    ha : Not (Membership.mem s a)
    ⊢ Disjoint v a
  -/
  rw [mem_compression] at h
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    h : Or (And (Membership.mem s a) (Membership.mem s (UV.compress u v a))) (And  …
    ha : Not (Membership.mem s a)
    ⊢ Disjoint v a
  -/
  obtain h | ⟨-, b, hb, hba⟩ := h
    /-
      case inl
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      h : And (Membership.mem s a) (Membership.mem s (UV.compress u v a))
      ⊢ Disjoint v a
    -/
  · cases ha h.1
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.intro.intro
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Not (Membership.mem s a)
    b : α
    hb : Membership.mem s b
    hba : Eq (UV.compress u v b) a
    ⊢ Disjoint v a
  -/
  unfold compress at hba
  /-
    case inr.intro.intro.intro
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Not (Membership.mem s a)
    b : α
    hb : Membership.mem s b
    hba : Eq (ite (And (Disjoint u b) (LE.le v b)) (SDiff.sdiff (Max.max b u) v) b …
    ⊢ Disjoint v a
  -/
  split_ifs at hba
    /-
      case pos
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      b : α
      hb : Membership.mem s b
      h✝ : And (Disjoint u b) (LE.le v b)
      hba : Eq (SDiff.sdiff (Max.max b u) v) a
      ⊢ Disjoint v a
    -/
  · rw [← hba]
    /-
      case pos
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      b : α
      hb : Membership.mem s b
      h✝ : And (Disjoint u b) (LE.le v b)
      hba : Eq (SDiff.sdiff (Max.max b u) v) a
      ⊢ Disjoint v (SDiff.sdiff (Max.max b u) v)
    -/
    exact disjoint_sdiff_self_right
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      b : α
      hb : Membership.mem s b
      h✝ : Not (And (Disjoint u b) (LE.le v b))
      hba : Eq b a
      ⊢ Disjoint v a
    -/
  · cases ne_of_mem_of_not_mem hb ha hba
    /-
      🎉 no goals
    -/


theorem sup_sdiff_mem_of_mem_compression_of_not_mem (h : a ∈ 𝓒 u v s) (ha : a ∉ s) :
    (a ⊔ v) \ u ∈ s := by
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    h : Membership.mem (UV.compression u v s) a
    ha : Not (Membership.mem s a)
    ⊢ Membership.mem s (SDiff.sdiff (Max.max a v) u)
  -/
  rw [mem_compression] at h
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    h : Or (And (Membership.mem s a) (Membership.mem s (UV.compress u v a))) (And  …
    ha : Not (Membership.mem s a)
    ⊢ Membership.mem s (SDiff.sdiff (Max.max a v) u)
  -/
  obtain h | ⟨-, b, hb, hba⟩ := h
    /-
      case inl
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      h : And (Membership.mem s a) (Membership.mem s (UV.compress u v a))
      ⊢ Membership.mem s (SDiff.sdiff (Max.max a v) u)
    -/
  · cases ha h.1
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.intro.intro
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Not (Membership.mem s a)
    b : α
    hb : Membership.mem s b
    hba : Eq (UV.compress u v b) a
    ⊢ Membership.mem s (SDiff.sdiff (Max.max a v) u)
  -/
  unfold compress at hba
  /-
    case inr.intro.intro.intro
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Not (Membership.mem s a)
    b : α
    hb : Membership.mem s b
    hba : Eq (ite (And (Disjoint u b) (LE.le v b)) (SDiff.sdiff (Max.max b u) v) b …
    ⊢ Membership.mem s (SDiff.sdiff (Max.max a v) u)
  -/
  split_ifs at hba with h
  · rwa [← hba, sdiff_sup_cancel (le_sup_of_le_left h.2), sup_sdiff_right_self,
      h.1.symm.sdiff_eq_left]
    /-
      case neg
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      ha : Not (Membership.mem s a)
      b : α
      hb : Membership.mem s b
      h : Not (And (Disjoint u b) (LE.le v b))
      hba : Eq b a
      ⊢ Membership.mem s (SDiff.sdiff (Max.max a v) u)
    -/
  · cases ne_of_mem_of_not_mem hb ha hba
    /-
      🎉 no goals
    -/


/-- If `a` is in the family compression and can be compressed, then its compression is in the
original family. -/
theorem sup_sdiff_mem_of_mem_compression (ha : a ∈ 𝓒 u v s) (hva : v ≤ a) (hua : Disjoint u a) :
    (a ⊔ u) \ v ∈ s := by
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Membership.mem (UV.compression u v s) a
    hva : LE.le v a
    hua : Disjoint u a
    ⊢ Membership.mem s (SDiff.sdiff (Max.max a u) v)
  -/
  rw [mem_compression, compress_of_disjoint_of_le hua hva] at ha
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Or (And (Membership.mem s a) (Membership.mem s (SDiff.sdiff (Max.max a u) …
    hva : LE.le v a
    hua : Disjoint u a
    ⊢ Membership.mem s (SDiff.sdiff (Max.max a u) v)
  -/
  obtain ⟨_, ha⟩ | ⟨_, b, hb, rfl⟩ := ha
    /-
      case inl.intro
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      hva : LE.le v a
      hua : Disjoint u a
      left✝ : Membership.mem s a
      ha : Membership.mem s (SDiff.sdiff (Max.max a u) v)
      ⊢ Membership.mem s (SDiff.sdiff (Max.max a u) v)
    -/
  · exact ha
    /-
      🎉 no goals
    -/
  have hu : u = ⊥ := by
    suffices Disjoint u (u \ v) by rwa [(hua.mono_right hva).sdiff_eq_left, disjoint_self] at this
    refine hua.mono_right ?_
    rw [← compress_idem, compress_of_disjoint_of_le hua hva]
    exact sdiff_le_sdiff_right le_sup_right
  have hv : v = ⊥ := by
    rw [← disjoint_self]
    apply Disjoint.mono_right hva
    rw [← compress_idem, compress_of_disjoint_of_le hua hva]
    exact disjoint_sdiff_self_right
  /-
    case inr.intro.intro.intro
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v : α
    inst✝ : DecidableEq α
    b : α
    hb : Membership.mem s b
    hva : LE.le v (UV.compress u v b)
    hua : Disjoint u (UV.compress u v b)
    left✝ : Not (Membership.mem s (UV.compress u v b))
    hu : Eq u Bot.bot
    hv : Eq v Bot.bot
    ⊢ Membership.mem s (SDiff.sdiff (Max.max (UV.compress u v b) u) v)
  -/
  rwa [hu, hv, compress_self, sup_bot_eq, sdiff_bot]
  /-
    🎉 no goals
  -/


/-- If `a` is in the `u, v`-compression but `v ≤ a`, then `a` must have been in the original
family. -/
theorem mem_of_mem_compression (ha : a ∈ 𝓒 u v s) (hva : v ≤ a) (hvu : v = ⊥ → u = ⊥) :
    a ∈ s := by
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Membership.mem (UV.compression u v s) a
    hva : LE.le v a
    hvu : Eq v Bot.bot → Eq u Bot.bot
    ⊢ Membership.mem s a
  -/
  rw [mem_compression] at ha
  /-
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    ha : Or (And (Membership.mem s a) (Membership.mem s (UV.compress u v a))) (And …
    hva : LE.le v a
    hvu : Eq v Bot.bot → Eq u Bot.bot
    ⊢ Membership.mem s a
  -/
  obtain ha | ⟨_, b, hb, h⟩ := ha
    /-
      case inl
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      hva : LE.le v a
      hvu : Eq v Bot.bot → Eq u Bot.bot
      ha : And (Membership.mem s a) (Membership.mem s (UV.compress u v a))
      ⊢ Membership.mem s a
    -/
  · exact ha.1
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.intro.intro
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    hva : LE.le v a
    hvu : Eq v Bot.bot → Eq u Bot.bot
    left✝ : Not (Membership.mem s a)
    b : α
    hb : Membership.mem s b
    h : Eq (UV.compress u v b) a
    ⊢ Membership.mem s a
  -/
  unfold compress at h
  /-
    case inr.intro.intro.intro
    α : Type u_1
    inst✝³ : GeneralizedBooleanAlgebra α
    inst✝² : DecidableRel Disjoint
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    s : Finset α
    u v a : α
    inst✝ : DecidableEq α
    hva : LE.le v a
    hvu : Eq v Bot.bot → Eq u Bot.bot
    left✝ : Not (Membership.mem s a)
    b : α
    hb : Membership.mem s b
    h : Eq (ite (And (Disjoint u b) (LE.le v b)) (SDiff.sdiff (Max.max b u) v) b) a
    ⊢ Membership.mem s a
  -/
  split_ifs at h
    /-
      case pos
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      hva : LE.le v a
      hvu : Eq v Bot.bot → Eq u Bot.bot
      left✝ : Not (Membership.mem s a)
      b : α
      hb : Membership.mem s b
      h✝ : And (Disjoint u b) (LE.le v b)
      h : Eq (SDiff.sdiff (Max.max b u) v) a
      ⊢ Membership.mem s a
    -/
  · rw [← h, le_sdiff_iff] at hva
    /-
      case pos
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      hvu : Eq v Bot.bot → Eq u Bot.bot
      left✝ : Not (Membership.mem s a)
      b : α
      hva : Eq v Bot.bot
      hb : Membership.mem s b
      h✝ : And (Disjoint u b) (LE.le v b)
      h : Eq (SDiff.sdiff (Max.max b u) v) a
      ⊢ Membership.mem s a
    -/
    rwa [← h, hvu hva, hva, sup_bot_eq, sdiff_bot]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝³ : GeneralizedBooleanAlgebra α
      inst✝² : DecidableRel Disjoint
      inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : Finset α
      u v a : α
      inst✝ : DecidableEq α
      hva : LE.le v a
      hvu : Eq v Bot.bot → Eq u Bot.bot
      left✝ : Not (Membership.mem s a)
      b : α
      hb : Membership.mem s b
      h✝ : Not (And (Disjoint u b) (LE.le v b))
      h : Eq b a
      ⊢ Membership.mem s a
    -/
  · rwa [← h]
    /-
      🎉 no goals
    -/


/-- Compressing a finset doesn't change its size. -/
theorem card_compress (huv : #u = #v) (a : Finset α) : #(compress u v a) = #a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    u v : Finset α
    huv : Eq u.card v.card
    a : Finset α
    ⊢ Eq (UV.compress u v a).card a.card
  -/
  unfold compress
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    u v : Finset α
    huv : Eq u.card v.card
    a : Finset α
    ⊢ Eq (ite (And (Disjoint u a) (LE.le v a)) (SDiff.sdiff (Max.max a u) v) a).ca …
  -/
  split_ifs with h
  · rw [card_sdiff (h.2.trans le_sup_left), sup_eq_union, card_union_of_disjoint h.1.symm, huv,
      add_tsub_cancel_right]
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      u v : Finset α
      huv : Eq u.card v.card
      a : Finset α
      h : Not (And (Disjoint u a) (LE.le v a))
      ⊢ Eq a.card a.card
    -/
  · rfl
    /-
      🎉 no goals
    -/


lemma _root_.Set.Sized.uvCompression (huv : #u = #v) (h𝒜 : (𝒜 : Set (Finset α)).Sized r) :
    (𝓒 u v 𝒜 : Set (Finset α)).Sized r := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    r : Nat
    huv : Eq u.card v.card
    h𝒜 : Set.Sized r ↑𝒜
    ⊢ Set.Sized r ↑(UV.compression u v 𝒜)
  -/
  simp_rw [Set.Sized, mem_coe, mem_compression]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    r : Nat
    huv : Eq u.card v.card
    h𝒜 : Set.Sized r ↑𝒜
    ⊢ ∀ ⦃x : Finset α⦄, Or (And (Membership.mem 𝒜 x) (Membership.mem 𝒜 (UV.compres …
  -/
  rintro s (hs | ⟨huvt, t, ht, rfl⟩)
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      u v : Finset α
      r : Nat
      huv : Eq u.card v.card
      h𝒜 : Set.Sized r ↑𝒜
      s : Finset α
      hs : And (Membership.mem 𝒜 s) (Membership.mem 𝒜 (UV.compress u v s))
      ⊢ Eq s.card r
    -/
  · exact h𝒜 hs.1
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      u v : Finset α
      r : Nat
      huv : Eq u.card v.card
      h𝒜 : Set.Sized r ↑𝒜
      t : Finset α
      ht : Membership.mem 𝒜 t
      huvt : Not (Membership.mem 𝒜 (UV.compress u v t))
      ⊢ Eq (UV.compress u v t).card r
    -/
  · rw [card_compress huv, h𝒜 ht]
    /-
      🎉 no goals
    -/


private theorem aux (huv : ∀ x ∈ u, ∃ y ∈ v, IsCompressed (u.erase x) (v.erase y) 𝒜) :
    v = ∅ → u = ∅ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    ⊢ Eq v EmptyCollection.emptyCollection → Eq u EmptyCollection.emptyCollection
  -/
  rintro rfl; refine eq_empty_of_forall_not_mem fun a ha ↦ ?_; obtain ⟨_, ⟨⟩, -⟩ := huv a ha
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- UV-compression reduces the size of the shadow of `𝒜` if, for all `x ∈ u` there is `y ∈ v` such
that `𝒜` is `(u.erase x, v.erase y)`-compressed. This is the key fact about compression for
Kruskal-Katona. -/
theorem shadow_compression_subset_compression_shadow (u v : Finset α)
    (huv : ∀ x ∈ u, ∃ y ∈ v, IsCompressed (u.erase x) (v.erase y) 𝒜) :
    ∂ (𝓒 u v 𝒜) ⊆ 𝓒 u v (∂ 𝒜) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    ⊢ HasSubset.Subset (UV.compression u v 𝒜).shadow (UV.compression u v 𝒜.shadow)
  -/
  set 𝒜' := 𝓒 u v 𝒜
  suffices H : ∀ s ∈ ∂ 𝒜',
      s ∉ ∂ 𝒜 → u ⊆ s ∧ Disjoint v s ∧ (s ∪ v) \ u ∈ ∂ 𝒜 ∧ (s ∪ v) \ u ∉ ∂ 𝒜' by
    rintro s hs'
    rw [mem_compression]
    by_cases hs : s ∈ 𝒜.shadow
    swap
    · obtain ⟨hus, hvs, h, _⟩ := H _ hs' hs
      exact Or.inr ⟨hs, _, h, compress_of_disjoint_of_le' hvs hus⟩
    refine Or.inl ⟨hs, ?_⟩
    rw [compress]
    split_ifs with huvs
    swap
    · exact hs
    rw [mem_shadow_iff] at hs'
    obtain ⟨t, Ht, a, hat, rfl⟩ := hs'
    have hav : a ∉ v := not_mem_mono huvs.2 (not_mem_erase a t)
    have hvt : v ≤ t := huvs.2.trans (erase_subset _ t)
    have ht : t ∈ 𝒜 := mem_of_mem_compression Ht hvt (aux huv)
    by_cases hau : a ∈ u
    · obtain ⟨b, hbv, Hcomp⟩ := huv a hau
      refine mem_shadow_iff_insert_mem.2 ⟨b, not_mem_sdiff_of_mem_right hbv, ?_⟩
      rw [← Hcomp.eq] at ht
      have hsb :=
        sup_sdiff_mem_of_mem_compression ht ((erase_subset _ _).trans hvt)
          (disjoint_erase_comm.2 huvs.1)
      rwa [sup_eq_union, sdiff_erase (mem_union_left _ <| hvt hbv), union_erase_of_mem hat, ←
        erase_union_of_mem hau] at hsb
    · refine mem_shadow_iff.2
        ⟨(t ⊔ u) \ v,
          sup_sdiff_mem_of_mem_compression Ht hvt <| disjoint_of_erase_right hau huvs.1, a, ?_, ?_⟩
      · rw [sup_eq_union, mem_sdiff, mem_union]
        exact ⟨Or.inl hat, hav⟩
      · rw [← erase_sdiff_comm, sup_eq_union, erase_union_distrib, erase_eq_of_not_mem hau]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    ⊢ ∀ (s : Finset α), Membership.mem 𝒜'.shadow s → Not (Membership.mem 𝒜.shadow  …
  -/
  intro s hs𝒜' hs𝒜
  -- This is going to be useful a couple of times so let's name it.
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Membership.mem 𝒜.shadow …
  -/
  have m : ∀ y, y ∉ s → insert y s ∉ 𝒜 := fun y h a => hs𝒜 (mem_shadow_iff_insert_mem.2 ⟨y, h, a⟩)
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Membership.mem 𝒜.shadow …
  -/
  obtain ⟨x, _, _⟩ := mem_shadow_iff_insert_mem.1 hs𝒜'
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Membership.mem 𝒜.shadow …
  -/
  have hus : u ⊆ insert x s := le_of_mem_compression_of_not_mem ‹_ ∈ 𝒜'› (m _ ‹x ∉ s›)
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus : HasSubset.Subset u (Insert.insert x s)
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Membership.mem 𝒜.shadow …
  -/
  have hvs : Disjoint v (insert x s) := disjoint_of_mem_compression_of_not_mem ‹_› (m _ ‹x ∉ s›)
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Membership.mem 𝒜.shadow …
  -/
  have : (insert x s ∪ v) \ u ∈ 𝒜 := sup_sdiff_mem_of_mem_compression_of_not_mem ‹_› (m _ ‹x ∉ s›)
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Membership.mem 𝒜.shadow …
  -/
  have hsv : Disjoint s v := hvs.symm.mono_left (subset_insert _ _)
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    hsv : Disjoint s v
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Membership.mem 𝒜.shadow …
  -/
  have hvu : Disjoint v u := disjoint_of_subset_right hus hvs
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    hsv : Disjoint s v
    hvu : Disjoint v u
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Membership.mem 𝒜.shadow …
  -/
  have hxv : x ∉ v := disjoint_right.1 hvs (mem_insert_self _ _)
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    hsv : Disjoint s v
    hvu : Disjoint v u
    hxv : Not (Membership.mem v x)
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Membership.mem 𝒜.shadow …
  -/
  have : v \ u = v := ‹Disjoint v u›.sdiff_eq_left
  -- The first key part is that `x ∉ u`
  have : x ∉ u := by
    intro hxu
    obtain ⟨y, hyv, hxy⟩ := huv x hxu
    -- If `x ∈ u`, we can get `y ∈ v` so that `𝒜` is `(u.erase x, v.erase y)`-compressed
    apply m y (disjoint_right.1 hsv hyv)
    -- and we will use this `y` to contradict `m`, so we would like to show `insert y s ∈ 𝒜`.
    -- We do this by showing the below
    have : ((insert x s ∪ v) \ u ∪ erase u x) \ erase v y ∈ 𝒜 := by
      refine
        sup_sdiff_mem_of_mem_compression (by rwa [hxy.eq]) ?_
          (disjoint_of_subset_left (erase_subset _ _) disjoint_sdiff)
      rw [union_sdiff_distrib, ‹v \ u = v›]
      exact (erase_subset _ _).trans subset_union_right
    -- and then arguing that it's the same
    convert this using 1
    rw [sdiff_union_erase_cancel (hus.trans subset_union_left) ‹x ∈ u›, erase_union_distrib,
      erase_insert ‹x ∉ s›, erase_eq_of_not_mem ‹x ∉ v›, sdiff_erase (mem_union_right _ hyv),
      union_sdiff_cancel_right hsv]
  -- Now that this is done, it's immediate that `u ⊆ s`
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this✝¹ : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    hsv : Disjoint s v
    hvu : Disjoint v u
    hxv : Not (Membership.mem v x)
    this✝ : Eq (SDiff.sdiff v u) v
    this : Not (Membership.mem u x)
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Membership.mem 𝒜.shadow …
  -/
  have hus : u ⊆ s := by rwa [← erase_eq_of_not_mem ‹x ∉ u›, ← subset_insert_iff]
  -- and we already had that `v` and `s` are disjoint,
  -- so it only remains to get `(s ∪ v) \ u ∈ ∂ 𝒜 \ ∂ 𝒜'`
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus✝ : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this✝¹ : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    hsv : Disjoint s v
    hvu : Disjoint v u
    hxv : Not (Membership.mem v x)
    this✝ : Eq (SDiff.sdiff v u) v
    this : Not (Membership.mem u x)
    hus : HasSubset.Subset u s
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Membership.mem 𝒜.shadow …
  -/
  simp_rw [mem_shadow_iff_insert_mem]
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus✝ : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this✝¹ : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    hsv : Disjoint s v
    hvu : Disjoint v u
    hxv : Not (Membership.mem v x)
    this✝ : Eq (SDiff.sdiff v u) v
    this : Not (Membership.mem u x)
    hus : HasSubset.Subset u s
    ⊢ And (HasSubset.Subset u s) (And (Disjoint v s) (And (Exists fun a => And (No …
  -/
  refine ⟨hus, hsv.symm, ⟨x, ?_, ?_⟩, ?_⟩
  -- `(s ∪ v) \ u ∈ ∂ 𝒜` is pretty direct:
    /-
      case intro.intro.refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      u v : Finset α
      huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
      𝒜' : Finset (Finset α) := UV.compression u v 𝒜
      s : Finset α
      hs𝒜' : Membership.mem 𝒜'.shadow s
      hs𝒜 : Not (Membership.mem 𝒜.shadow s)
      m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
      x : α
      left✝ : Not (Membership.mem s x)
      right✝ : Membership.mem 𝒜' (Insert.insert x s)
      hus✝ : HasSubset.Subset u (Insert.insert x s)
      hvs : Disjoint v (Insert.insert x s)
      this✝¹ : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
      hsv : Disjoint s v
      hvu : Disjoint v u
      hxv : Not (Membership.mem v x)
      this✝ : Eq (SDiff.sdiff v u) v
      this : Not (Membership.mem u x)
      hus : HasSubset.Subset u s
      ⊢ Not (Membership.mem (SDiff.sdiff (Union.union s v) u) x)
    -/
  · exact not_mem_sdiff_of_not_mem_left (not_mem_union.2 ⟨‹x ∉ s›, ‹x ∉ v›⟩)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      u v : Finset α
      huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
      𝒜' : Finset (Finset α) := UV.compression u v 𝒜
      s : Finset α
      hs𝒜' : Membership.mem 𝒜'.shadow s
      hs𝒜 : Not (Membership.mem 𝒜.shadow s)
      m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
      x : α
      left✝ : Not (Membership.mem s x)
      right✝ : Membership.mem 𝒜' (Insert.insert x s)
      hus✝ : HasSubset.Subset u (Insert.insert x s)
      hvs : Disjoint v (Insert.insert x s)
      this✝¹ : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
      hsv : Disjoint s v
      hvu : Disjoint v u
      hxv : Not (Membership.mem v x)
      this✝ : Eq (SDiff.sdiff v u) v
      this : Not (Membership.mem u x)
      hus : HasSubset.Subset u s
      ⊢ Membership.mem 𝒜 (Insert.insert x (SDiff.sdiff (Union.union s v) u))
    -/
  · rwa [← insert_sdiff_of_not_mem _ ‹x ∉ u›, ← insert_union]
    /-
      🎉 no goals
    -/
  -- For (s ∪ v) \ u ∉ ∂ 𝒜', we split up based on w ∈ u
  /-
    case intro.intro.refine_3
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus✝ : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this✝¹ : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    hsv : Disjoint s v
    hvu : Disjoint v u
    hxv : Not (Membership.mem v x)
    this✝ : Eq (SDiff.sdiff v u) v
    this : Not (Membership.mem u x)
    hus : HasSubset.Subset u s
    ⊢ Not (Exists fun a => And (Not (Membership.mem (SDiff.sdiff (Union.union s v) …
  -/
  rintro ⟨w, hwB, hw𝒜'⟩
  have : v ⊆ insert w ((s ∪ v) \ u) :=
    (subset_sdiff.2 ⟨subset_union_right, hvu⟩).trans (subset_insert _ _)
  /-
    case intro.intro.refine_3.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus✝ : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this✝² : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    hsv : Disjoint s v
    hvu : Disjoint v u
    hxv : Not (Membership.mem v x)
    this✝¹ : Eq (SDiff.sdiff v u) v
    this✝ : Not (Membership.mem u x)
    hus : HasSubset.Subset u s
    w : α
    hwB : Not (Membership.mem (SDiff.sdiff (Union.union s v) u) w)
    hw𝒜' : Membership.mem 𝒜' (Insert.insert w (SDiff.sdiff (Union.union s v) u))
    this : HasSubset.Subset v (Insert.insert w (SDiff.sdiff (Union.union s v) u))
    ⊢ False
  -/
  by_cases hwu : w ∈ u
  -- If `w ∈ u`, we find `z ∈ v`, and contradict `m` again
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      u v : Finset α
      huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
      𝒜' : Finset (Finset α) := UV.compression u v 𝒜
      s : Finset α
      hs𝒜' : Membership.mem 𝒜'.shadow s
      hs𝒜 : Not (Membership.mem 𝒜.shadow s)
      m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
      x : α
      left✝ : Not (Membership.mem s x)
      right✝ : Membership.mem 𝒜' (Insert.insert x s)
      hus✝ : HasSubset.Subset u (Insert.insert x s)
      hvs : Disjoint v (Insert.insert x s)
      this✝² : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
      hsv : Disjoint s v
      hvu : Disjoint v u
      hxv : Not (Membership.mem v x)
      this✝¹ : Eq (SDiff.sdiff v u) v
      this✝ : Not (Membership.mem u x)
      hus : HasSubset.Subset u s
      w : α
      hwB : Not (Membership.mem (SDiff.sdiff (Union.union s v) u) w)
      hw𝒜' : Membership.mem 𝒜' (Insert.insert w (SDiff.sdiff (Union.union s v) u))
      this : HasSubset.Subset v (Insert.insert w (SDiff.sdiff (Union.union s v) u))
      hwu : Membership.mem u w
      ⊢ False
    -/
  · obtain ⟨z, hz, hxy⟩ := huv w hwu
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      u v : Finset α
      huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
      𝒜' : Finset (Finset α) := UV.compression u v 𝒜
      s : Finset α
      hs𝒜' : Membership.mem 𝒜'.shadow s
      hs𝒜 : Not (Membership.mem 𝒜.shadow s)
      m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
      x : α
      left✝ : Not (Membership.mem s x)
      right✝ : Membership.mem 𝒜' (Insert.insert x s)
      hus✝ : HasSubset.Subset u (Insert.insert x s)
      hvs : Disjoint v (Insert.insert x s)
      this✝² : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
      hsv : Disjoint s v
      hvu : Disjoint v u
      hxv : Not (Membership.mem v x)
      this✝¹ : Eq (SDiff.sdiff v u) v
      this✝ : Not (Membership.mem u x)
      hus : HasSubset.Subset u s
      w : α
      hwB : Not (Membership.mem (SDiff.sdiff (Union.union s v) u) w)
      hw𝒜' : Membership.mem 𝒜' (Insert.insert w (SDiff.sdiff (Union.union s v) u))
      this : HasSubset.Subset v (Insert.insert w (SDiff.sdiff (Union.union s v) u))
      hwu : Membership.mem u w
      z : α
      hz : Membership.mem v z
      hxy : UV.IsCompressed (u.erase w) (v.erase z) 𝒜
      ⊢ False
    -/
    apply m z (disjoint_right.1 hsv hz)
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      u v : Finset α
      huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
      𝒜' : Finset (Finset α) := UV.compression u v 𝒜
      s : Finset α
      hs𝒜' : Membership.mem 𝒜'.shadow s
      hs𝒜 : Not (Membership.mem 𝒜.shadow s)
      m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
      x : α
      left✝ : Not (Membership.mem s x)
      right✝ : Membership.mem 𝒜' (Insert.insert x s)
      hus✝ : HasSubset.Subset u (Insert.insert x s)
      hvs : Disjoint v (Insert.insert x s)
      this✝² : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
      hsv : Disjoint s v
      hvu : Disjoint v u
      hxv : Not (Membership.mem v x)
      this✝¹ : Eq (SDiff.sdiff v u) v
      this✝ : Not (Membership.mem u x)
      hus : HasSubset.Subset u s
      w : α
      hwB : Not (Membership.mem (SDiff.sdiff (Union.union s v) u) w)
      hw𝒜' : Membership.mem 𝒜' (Insert.insert w (SDiff.sdiff (Union.union s v) u))
      this : HasSubset.Subset v (Insert.insert w (SDiff.sdiff (Union.union s v) u))
      hwu : Membership.mem u w
      z : α
      hz : Membership.mem v z
      hxy : UV.IsCompressed (u.erase w) (v.erase z) 𝒜
      ⊢ Membership.mem 𝒜 (Insert.insert z s)
    -/
    have : insert w ((s ∪ v) \ u) ∈ 𝒜 := mem_of_mem_compression hw𝒜' ‹_› (aux huv)
    have : (insert w ((s ∪ v) \ u) ∪ erase u w) \ erase v z ∈ 𝒜 := by
      refine sup_sdiff_mem_of_mem_compression (by rwa [hxy.eq]) ((erase_subset _ _).trans ‹_›) ?_
      rw [← sdiff_erase (mem_union_left _ <| hus hwu)]
      exact disjoint_sdiff
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      u v : Finset α
      huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
      𝒜' : Finset (Finset α) := UV.compression u v 𝒜
      s : Finset α
      hs𝒜' : Membership.mem 𝒜'.shadow s
      hs𝒜 : Not (Membership.mem 𝒜.shadow s)
      m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
      x : α
      left✝ : Not (Membership.mem s x)
      right✝ : Membership.mem 𝒜' (Insert.insert x s)
      hus✝ : HasSubset.Subset u (Insert.insert x s)
      hvs : Disjoint v (Insert.insert x s)
      this✝⁴ : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
      hsv : Disjoint s v
      hvu : Disjoint v u
      hxv : Not (Membership.mem v x)
      this✝³ : Eq (SDiff.sdiff v u) v
      this✝² : Not (Membership.mem u x)
      hus : HasSubset.Subset u s
      w : α
      hwB : Not (Membership.mem (SDiff.sdiff (Union.union s v) u) w)
      hw𝒜' : Membership.mem 𝒜' (Insert.insert w (SDiff.sdiff (Union.union s v) u))
      this✝¹ : HasSubset.Subset v (Insert.insert w (SDiff.sdiff (Union.union s v) u))
      hwu : Membership.mem u w
      z : α
      hz : Membership.mem v z
      hxy : UV.IsCompressed (u.erase w) (v.erase z) 𝒜
      this✝ : Membership.mem 𝒜 (Insert.insert w (SDiff.sdiff (Union.union s v) u))
      this : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert w (SDiff.sdif …
      ⊢ Membership.mem 𝒜 (Insert.insert z s)
    -/
    convert this using 1
    rw [insert_union_comm, insert_erase ‹w ∈ u›,
      sdiff_union_of_subset (hus.trans subset_union_left),
      sdiff_erase (mem_union_right _ ‹z ∈ v›), union_sdiff_cancel_right hsv]
  -- If `w ∉ u`, we contradict `m` again
  /-
    case neg
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus✝ : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this✝² : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    hsv : Disjoint s v
    hvu : Disjoint v u
    hxv : Not (Membership.mem v x)
    this✝¹ : Eq (SDiff.sdiff v u) v
    this✝ : Not (Membership.mem u x)
    hus : HasSubset.Subset u s
    w : α
    hwB : Not (Membership.mem (SDiff.sdiff (Union.union s v) u) w)
    hw𝒜' : Membership.mem 𝒜' (Insert.insert w (SDiff.sdiff (Union.union s v) u))
    this : HasSubset.Subset v (Insert.insert w (SDiff.sdiff (Union.union s v) u))
    hwu : Not (Membership.mem u w)
    ⊢ False
  -/
  rw [mem_sdiff, ← Classical.not_imp, Classical.not_not] at hwB
  /-
    case neg
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus✝ : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this✝² : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    hsv : Disjoint s v
    hvu : Disjoint v u
    hxv : Not (Membership.mem v x)
    this✝¹ : Eq (SDiff.sdiff v u) v
    this✝ : Not (Membership.mem u x)
    hus : HasSubset.Subset u s
    w : α
    hwB : Membership.mem (Union.union s v) w → Membership.mem u w
    hw𝒜' : Membership.mem 𝒜' (Insert.insert w (SDiff.sdiff (Union.union s v) u))
    this : HasSubset.Subset v (Insert.insert w (SDiff.sdiff (Union.union s v) u))
    hwu : Not (Membership.mem u w)
    ⊢ False
  -/
  apply m w (hwu ∘ hwB ∘ mem_union_left _)
  have : (insert w ((s ∪ v) \ u) ∪ u) \ v ∈ 𝒜 :=
    sup_sdiff_mem_of_mem_compression ‹insert w ((s ∪ v) \ u) ∈ 𝒜'› ‹_›
      (disjoint_insert_right.2 ⟨‹_›, disjoint_sdiff⟩)
  /-
    case neg
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    u v : Finset α
    huv : ∀ (x : α), Membership.mem u x → Exists fun y => And (Membership.mem v y) …
    𝒜' : Finset (Finset α) := UV.compression u v 𝒜
    s : Finset α
    hs𝒜' : Membership.mem 𝒜'.shadow s
    hs𝒜 : Not (Membership.mem 𝒜.shadow s)
    m : ∀ (y : α), Not (Membership.mem s y) → Not (Membership.mem 𝒜 (Insert.insert …
    x : α
    left✝ : Not (Membership.mem s x)
    right✝ : Membership.mem 𝒜' (Insert.insert x s)
    hus✝ : HasSubset.Subset u (Insert.insert x s)
    hvs : Disjoint v (Insert.insert x s)
    this✝³ : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert x s) v) u)
    hsv : Disjoint s v
    hvu : Disjoint v u
    hxv : Not (Membership.mem v x)
    this✝² : Eq (SDiff.sdiff v u) v
    this✝¹ : Not (Membership.mem u x)
    hus : HasSubset.Subset u s
    w : α
    hwB : Membership.mem (Union.union s v) w → Membership.mem u w
    hw𝒜' : Membership.mem 𝒜' (Insert.insert w (SDiff.sdiff (Union.union s v) u))
    this✝ : HasSubset.Subset v (Insert.insert w (SDiff.sdiff (Union.union s v) u))
    hwu : Not (Membership.mem u w)
    this : Membership.mem 𝒜 (SDiff.sdiff (Union.union (Insert.insert w (SDiff.sdif …
    ⊢ Membership.mem 𝒜 (Insert.insert w s)
  -/
  convert this using 1
  rw [insert_union, sdiff_union_of_subset (hus.trans subset_union_left),
    insert_sdiff_of_not_mem _ (hwu ∘ hwB ∘ mem_union_right _), union_sdiff_cancel_right hsv]


/-- UV-compression reduces the size of the shadow of `𝒜` if, for all `x ∈ u` there is `y ∈ v`
such that `𝒜` is `(u.erase x, v.erase y)`-compressed. This is the key UV-compression fact needed for
Kruskal-Katona. -/
theorem card_shadow_compression_le (u v : Finset α)
    (huv : ∀ x ∈ u, ∃ y ∈ v, IsCompressed (u.erase x) (v.erase y) 𝒜) :
    #(∂ (𝓒 u v 𝒜)) ≤ #(∂ 𝒜) :=
  (card_le_card <| shadow_compression_subset_compression_shadow _ _ huv).trans
    (card_compression _ _ _).le


