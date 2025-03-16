/-- The antidiagonal of a natural number `n` is
    the finset of pairs `(i, j)` such that `i + j = n`. -/
instance instHasAntidiagonal : HasAntidiagonal ℕ where
  antidiagonal n := ⟨Multiset.Nat.antidiagonal n, Multiset.Nat.nodup_antidiagonal n⟩
  mem_antidiagonal {n} {xy} := by
    /-
      n : Nat
      xy : Prod Nat Nat
      ⊢ Iff (Membership.mem ((fun n => { val := Multiset.Nat.antidiagonal n, nodup : …
    -/
    rw [mem_def, Multiset.Nat.mem_antidiagonal]
    /-
      🎉 no goals
    -/


lemma antidiagonal_eq_map (n : ℕ) :
    antidiagonal n = (range (n + 1)).map ⟨fun i ↦ (i, n - i), fun _ _ h ↦ (Prod.ext_iff.1 h).1⟩ :=
  rfl


lemma antidiagonal_eq_map' (n : ℕ) :
    antidiagonal n =
      (range (n + 1)).map ⟨fun i ↦ (n - i, i), fun _ _ h ↦ (Prod.ext_iff.1 h).2⟩ := by
  /-
    n : Nat
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal n) (Finset.map { toFun := fun i => { …
  -/
  rw [← map_swap_antidiagonal, antidiagonal_eq_map, map_map]; rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma antidiagonal_eq_image (n : ℕ) :
    antidiagonal n = (range (n + 1)).image fun i ↦ (i, n - i) := by
  /-
    n : Nat
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal n) (Finset.image (fun i => { fst :=  …
  -/
  simp only [antidiagonal_eq_map, map_eq_image, Function.Embedding.coeFn_mk]
  /-
    🎉 no goals
  -/


lemma antidiagonal_eq_image' (n : ℕ) :
    antidiagonal n = (range (n + 1)).image fun i ↦ (n - i, i) := by
  /-
    n : Nat
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal n) (Finset.image (fun i => { fst :=  …
  -/
  simp only [antidiagonal_eq_map', map_eq_image, Function.Embedding.coeFn_mk]
  /-
    🎉 no goals
  -/


/-- The cardinality of the antidiagonal of `n` is `n + 1`. -/
@[simp]
                                                                        /-
                                                                          n : Nat
                                                                          ⊢ Eq (Finset.HasAntidiagonal.antidiagonal n).card (HAdd.hAdd n 1)
                                                                        -/
theorem card_antidiagonal (n : ℕ) : (antidiagonal n).card = n + 1 := by simp [antidiagonal]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- The antidiagonal of `0` is the list `[(0, 0)]` -/
@[simp]
theorem antidiagonal_zero : antidiagonal 0 = {(0, 0)} := rfl


theorem antidiagonal_succ (n : ℕ) :
    antidiagonal (n + 1) =
      cons (0, n + 1)
        ((antidiagonal n).map
          (Embedding.prodMap ⟨Nat.succ, Nat.succ_injective⟩ (Embedding.refl _)))
            /-
              n : Nat
              ⊢ Not (Membership.mem (Finset.map ({ toFun := Nat.succ, inj' := Nat.succ_injec …
            -/
        (by simp) := by
            /-
              🎉 no goals
            -/
  /-
    n : Nat
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)) (Finset.cons { fst  …
  -/
  apply eq_of_veq
  /-
    case a
    n : Nat
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).val (Finset.cons {  …
  -/
  rw [cons_val, map_val]
  /-
    case a
    n : Nat
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).val (Multiset.cons  …
  -/
  apply Multiset.Nat.antidiagonal_succ
  /-
    🎉 no goals
  -/


theorem antidiagonal_succ' (n : ℕ) :
    antidiagonal (n + 1) =
      cons (n + 1, 0)
        ((antidiagonal n).map
          (Embedding.prodMap (Embedding.refl _) ⟨Nat.succ, Nat.succ_injective⟩))
            /-
              n : Nat
              ⊢ Not (Membership.mem (Finset.map ((Function.Embedding.refl Nat).prodMap { toF …
            -/
        (by simp) := by
            /-
              🎉 no goals
            -/
  /-
    n : Nat
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)) (Finset.cons { fst  …
  -/
  apply eq_of_veq
  /-
    case a
    n : Nat
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).val (Finset.cons {  …
  -/
  rw [cons_val, map_val]
  /-
    case a
    n : Nat
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).val (Multiset.cons  …
  -/
  exact Multiset.Nat.antidiagonal_succ'
  /-
    🎉 no goals
  -/


theorem antidiagonal_succ_succ' {n : ℕ} :
    antidiagonal (n + 2) =
      cons (0, n + 2)
        (cons (n + 2, 0)
            ((antidiagonal n).map
              (Embedding.prodMap ⟨Nat.succ, Nat.succ_injective⟩
                ⟨Nat.succ, Nat.succ_injective⟩)) <|
             /-
               n : Nat
               ⊢ Not (Membership.mem (Finset.map ({ toFun := Nat.succ, inj' := Nat.succ_injec …
             -/
          by simp)
             /-
               🎉 no goals
             -/
            /-
              n : Nat
              ⊢ Not (Membership.mem (Finset.cons { fst := HAdd.hAdd n 2, snd := 0 } (Finset. …
            -/
        (by simp) := by
            /-
              🎉 no goals
            -/
  /-
    n : Nat
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 2)) (Finset.cons { fst  …
  -/
  simp_rw [antidiagonal_succ (n + 1), antidiagonal_succ', Finset.map_cons, map_map]
  /-
    n : Nat
    ⊢ Eq (Finset.cons { fst := 0, snd := HAdd.hAdd (HAdd.hAdd n 1) 1 } (Finset.con …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem antidiagonal.fst_lt {n : ℕ} {kl : ℕ × ℕ} (hlk : kl ∈ antidiagonal n) : kl.1 < n + 1 :=
  Nat.lt_succ_of_le <| antidiagonal.fst_le hlk


theorem antidiagonal.snd_lt {n : ℕ} {kl : ℕ × ℕ} (hlk : kl ∈ antidiagonal n) : kl.2 < n + 1 :=
  Nat.lt_succ_of_le <| antidiagonal.snd_le hlk


@[simp] lemma antidiagonal_filter_snd_le_of_le {n k : ℕ} (h : k ≤ n) :
    (antidiagonal n).filter (fun a ↦ a.snd ≤ k) = (antidiagonal k).map
      (Embedding.prodMap ⟨_, add_left_injective (n - k)⟩ (Embedding.refl ℕ)) := by
  /-
    n k : Nat
    h : LE.le k n
    ⊢ Eq (Finset.filter (fun a => LE.le a.2 k) (Finset.HasAntidiagonal.antidiagona …
  -/
  ext ⟨i, j⟩
  /-
    case h.mk
    n k : Nat
    h : LE.le k n
    i j : Nat
    ⊢ Iff (Membership.mem (Finset.filter (fun a => LE.le a.2 k) (Finset.HasAntidia …
  -/
  suffices i + j = n ∧ j ≤ k ↔ ∃ a, a + j = k ∧ a + (n - k) = i by simpa
  /-
    case h.mk
    n k : Nat
    h : LE.le k n
    i j : Nat
    ⊢ Iff (And (Eq (HAdd.hAdd i j) n) (LE.le j k)) (Exists fun a => And (Eq (HAdd. …
  -/
  refine ⟨fun hi ↦ ⟨k - j, tsub_add_cancel_of_le hi.2, ?_⟩, ?_⟩
  · rw [add_comm, tsub_add_eq_add_tsub h, ← hi.1, add_assoc, Nat.add_sub_of_le hi.2,
      add_tsub_cancel_right]
    /-
      case h.mk.refine_2
      n k : Nat
      h : LE.le k n
      i j : Nat
      ⊢ (Exists fun a => And (Eq (HAdd.hAdd a j) k) (Eq (HAdd.hAdd a (HSub.hSub n k) …
    -/
  · rintro ⟨l, hl, rfl⟩
    /-
      case h.mk.refine_2.intro.intro
      n k : Nat
      h : LE.le k n
      j l : Nat
      hl : Eq (HAdd.hAdd l j) k
      ⊢ And (Eq (HAdd.hAdd (HAdd.hAdd l (HSub.hSub n k)) j) n) (LE.le j k)
    -/
    refine ⟨?_, hl ▸ Nat.le_add_left j l⟩
    /-
      case h.mk.refine_2.intro.intro
      n k : Nat
      h : LE.le k n
      j l : Nat
      hl : Eq (HAdd.hAdd l j) k
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd l (HSub.hSub n k)) j) n
    -/
    rw [add_assoc, add_comm, add_assoc, add_comm j l, hl]
    /-
      case h.mk.refine_2.intro.intro
      n k : Nat
      h : LE.le k n
      j l : Nat
      hl : Eq (HAdd.hAdd l j) k
      ⊢ Eq (HAdd.hAdd (HSub.hSub n k) k) n
    -/
    exact Nat.sub_add_cancel h
    /-
      🎉 no goals
    -/


@[simp] lemma antidiagonal_filter_fst_le_of_le {n k : ℕ} (h : k ≤ n) :
    (antidiagonal n).filter (fun a ↦ a.fst ≤ k) = (antidiagonal k).map
      (Embedding.prodMap (Embedding.refl ℕ) ⟨_, add_left_injective (n - k)⟩) := by
  /-
    n k : Nat
    h : LE.le k n
    ⊢ Eq (Finset.filter (fun a => LE.le a.1 k) (Finset.HasAntidiagonal.antidiagona …
  -/
  have aux₁ : fun a ↦ a.fst ≤ k = (fun a ↦ a.snd ≤ k) ∘ (Equiv.prodComm ℕ ℕ).symm := rfl
  have aux₂ : ∀ i j, (∃ a b, a + b = k ∧ b = i ∧ a + (n - k) = j) ↔
                      ∃ a b, a + b = k ∧ a = i ∧ b + (n - k) = j :=
    fun i j ↦ by rw [exists_comm]; exact exists₂_congr (fun a b ↦ by rw [add_comm])
  /-
    n k : Nat
    h : LE.le k n
    aux₁ : Eq (fun a => LE.le a.1 k) (Function.comp (fun a => LE.le a.2 k) ⇑(Equiv …
    aux₂ : ∀ (i j : Nat), Iff (Exists fun a => Exists fun b => And (Eq (HAdd.hAdd  …
    ⊢ Eq (Finset.filter (fun a => LE.le a.1 k) (Finset.HasAntidiagonal.antidiagona …
  -/
  rw [← map_prodComm_antidiagonal]
  /-
    n k : Nat
    h : LE.le k n
    aux₁ : Eq (fun a => LE.le a.1 k) (Function.comp (fun a => LE.le a.2 k) ⇑(Equiv …
    aux₂ : ∀ (i j : Nat), Iff (Exists fun a => Exists fun b => And (Eq (HAdd.hAdd  …
    ⊢ Eq (Finset.filter (fun a => LE.le a.1 k) (Finset.map (Equiv.prodComm Nat Nat …
  -/
  simp_rw [aux₁, ← map_filter, antidiagonal_filter_snd_le_of_le h, map_map]
  /-
    n k : Nat
    h : LE.le k n
    aux₁ : Eq (fun a => LE.le a.1 k) (Function.comp (fun a => LE.le a.2 k) ⇑(Equiv …
    aux₂ : ∀ (i j : Nat), Iff (Exists fun a => Exists fun b => And (Eq (HAdd.hAdd  …
    ⊢ Eq (Finset.map (({ toFun := fun x => HAdd.hAdd x (HSub.hSub n k), inj' := ⋯  …
  -/
  ext ⟨i, j⟩
  /-
    case h.mk
    n k : Nat
    h : LE.le k n
    aux₁ : Eq (fun a => LE.le a.1 k) (Function.comp (fun a => LE.le a.2 k) ⇑(Equiv …
    aux₂ : ∀ (i j : Nat), Iff (Exists fun a => Exists fun b => And (Eq (HAdd.hAdd  …
    i j : Nat
    ⊢ Iff (Membership.mem (Finset.map (({ toFun := fun x => HAdd.hAdd x (HSub.hSub …
  -/
  simpa using aux₂ i j
  /-
    🎉 no goals
  -/


@[simp] lemma antidiagonal_filter_le_fst_of_le {n k : ℕ} (h : k ≤ n) :
    (antidiagonal n).filter (fun a ↦ k ≤ a.fst) = (antidiagonal (n - k)).map
      (Embedding.prodMap ⟨_, add_left_injective k⟩ (Embedding.refl ℕ)) := by
  /-
    n k : Nat
    h : LE.le k n
    ⊢ Eq (Finset.filter (fun a => LE.le k a.1) (Finset.HasAntidiagonal.antidiagona …
  -/
  ext ⟨i, j⟩
  /-
    case h.mk
    n k : Nat
    h : LE.le k n
    i j : Nat
    ⊢ Iff (Membership.mem (Finset.filter (fun a => LE.le k a.1) (Finset.HasAntidia …
  -/
  suffices i + j = n ∧ k ≤ i ↔ ∃ a, a + j = n - k ∧ a + k = i by simpa
  /-
    case h.mk
    n k : Nat
    h : LE.le k n
    i j : Nat
    ⊢ Iff (And (Eq (HAdd.hAdd i j) n) (LE.le k i)) (Exists fun a => And (Eq (HAdd. …
  -/
  refine ⟨fun hi ↦ ⟨i - k, ?_, tsub_add_cancel_of_le hi.2⟩, ?_⟩
    /-
      case h.mk.refine_1
      n k : Nat
      h : LE.le k n
      i j : Nat
      hi : And (Eq (HAdd.hAdd i j) n) (LE.le k i)
      ⊢ Eq (HAdd.hAdd (HSub.hSub i k) j) (HSub.hSub n k)
    -/
  · rw [← Nat.sub_add_comm hi.2, hi.1]
    /-
      🎉 no goals
    -/
    /-
      case h.mk.refine_2
      n k : Nat
      h : LE.le k n
      i j : Nat
      ⊢ (Exists fun a => And (Eq (HAdd.hAdd a j) (HSub.hSub n k)) (Eq (HAdd.hAdd a k …
    -/
  · rintro ⟨l, hl, rfl⟩
    /-
      case h.mk.refine_2.intro.intro
      n k : Nat
      h : LE.le k n
      j l : Nat
      hl : Eq (HAdd.hAdd l j) (HSub.hSub n k)
      ⊢ And (Eq (HAdd.hAdd (HAdd.hAdd l k) j) n) (LE.le k (HAdd.hAdd l k))
    -/
    refine ⟨?_, Nat.le_add_left k l⟩
    /-
      case h.mk.refine_2.intro.intro
      n k : Nat
      h : LE.le k n
      j l : Nat
      hl : Eq (HAdd.hAdd l j) (HSub.hSub n k)
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd l k) j) n
    -/
    rw [add_right_comm, hl]
    /-
      case h.mk.refine_2.intro.intro
      n k : Nat
      h : LE.le k n
      j l : Nat
      hl : Eq (HAdd.hAdd l j) (HSub.hSub n k)
      ⊢ Eq (HAdd.hAdd (HSub.hSub n k) k) n
    -/
    exact tsub_add_cancel_of_le h
    /-
      🎉 no goals
    -/


@[simp] lemma antidiagonal_filter_le_snd_of_le {n k : ℕ} (h : k ≤ n) :
    (antidiagonal n).filter (fun a ↦ k ≤ a.snd) = (antidiagonal (n - k)).map
      (Embedding.prodMap (Embedding.refl ℕ) ⟨_, add_left_injective k⟩) := by
  /-
    n k : Nat
    h : LE.le k n
    ⊢ Eq (Finset.filter (fun a => LE.le k a.2) (Finset.HasAntidiagonal.antidiagona …
  -/
  have aux₁ : fun a ↦ k ≤ a.snd = (fun a ↦ k ≤ a.fst) ∘ (Equiv.prodComm ℕ ℕ).symm := rfl
  have aux₂ : ∀ i j, (∃ a b, a + b = n - k ∧ b = i ∧ a + k = j) ↔
                      ∃ a b, a + b = n - k ∧ a = i ∧ b + k = j :=
    fun i j ↦ by rw [exists_comm]; exact exists₂_congr (fun a b ↦ by rw [add_comm])
  /-
    n k : Nat
    h : LE.le k n
    aux₁ : Eq (fun a => LE.le k a.2) (Function.comp (fun a => LE.le k a.1) ⇑(Equiv …
    aux₂ : ∀ (i j : Nat), Iff (Exists fun a => Exists fun b => And (Eq (HAdd.hAdd  …
    ⊢ Eq (Finset.filter (fun a => LE.le k a.2) (Finset.HasAntidiagonal.antidiagona …
  -/
  rw [← map_prodComm_antidiagonal]
  /-
    n k : Nat
    h : LE.le k n
    aux₁ : Eq (fun a => LE.le k a.2) (Function.comp (fun a => LE.le k a.1) ⇑(Equiv …
    aux₂ : ∀ (i j : Nat), Iff (Exists fun a => Exists fun b => And (Eq (HAdd.hAdd  …
    ⊢ Eq (Finset.filter (fun a => LE.le k a.2) (Finset.map (Equiv.prodComm Nat Nat …
  -/
  simp_rw [aux₁, ← map_filter, antidiagonal_filter_le_fst_of_le h, map_map]
  /-
    n k : Nat
    h : LE.le k n
    aux₁ : Eq (fun a => LE.le k a.2) (Function.comp (fun a => LE.le k a.1) ⇑(Equiv …
    aux₂ : ∀ (i j : Nat), Iff (Exists fun a => Exists fun b => And (Eq (HAdd.hAdd  …
    ⊢ Eq (Finset.map (({ toFun := fun x => HAdd.hAdd x k, inj' := ⋯ }.prodMap (Fun …
  -/
  ext ⟨i, j⟩
  /-
    case h.mk
    n k : Nat
    h : LE.le k n
    aux₁ : Eq (fun a => LE.le k a.2) (Function.comp (fun a => LE.le k a.1) ⇑(Equiv …
    aux₂ : ∀ (i j : Nat), Iff (Exists fun a => Exists fun b => And (Eq (HAdd.hAdd  …
    i j : Nat
    ⊢ Iff (Membership.mem (Finset.map (({ toFun := fun x => HAdd.hAdd x k, inj' := …
  -/
  simpa using aux₂ i j
  /-
    🎉 no goals
  -/


/-- The set `antidiagonal n` is equivalent to `Fin (n+1)`, via the first projection. --/
@[simps]
def antidiagonalEquivFin (n : ℕ) : antidiagonal n ≃ Fin (n + 1) where
  toFun := fun ⟨⟨i, _⟩, h⟩ ↦ ⟨i, antidiagonal.fst_lt h⟩
  invFun := fun ⟨i, h⟩ ↦ ⟨⟨i, n - i⟩, by
    /-
      n : Nat
      x✝ : Fin (HAdd.hAdd n 1)
      i : Nat
      h : LT.lt i (HAdd.hAdd n 1)
      ⊢ Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd := HS …
    -/
    rw [mem_antidiagonal, add_comm, Nat.sub_add_cancel]
    /-
      n : Nat
      x✝ : Fin (HAdd.hAdd n 1)
      i : Nat
      h : LT.lt i (HAdd.hAdd n 1)
      ⊢ LE.le i n
    -/
    exact Nat.le_of_lt_succ h⟩
    /-
      🎉 no goals
    -/
                 /-
                   n : Nat
                   ⊢ Function.LeftInverse (fun x => Finset.Nat.antidiagonalEquivFin.match_2 n (fu …
                 -/
  left_inv := by rintro ⟨⟨i, j⟩, h⟩; ext; rfl
                                          /-
                                            🎉 no goals
                                          -/
  right_inv _ := rfl


