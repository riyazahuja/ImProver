                                                                                /-
                                                                                  α : Type u
                                                                                  n : Nat
                                                                                  ⊢ ∀ (a : Nat), Membership.mem (List.range n) a → LT.lt a n
                                                                                -/
theorem finRange_eq_pmap_range (n : ℕ) : finRange n = (range n).pmap Fin.mk (by simp) := by
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  /-
    n : Nat
    ⊢ Eq (List.finRange n) (List.pmap Fin.mk (List.range n) ⋯)
  -/
                             /-
                               🎉 no goals
                             -/
  apply List.ext_getElem <;> simp [finRange]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem mem_finRange {n : ℕ} (a : Fin n) : a ∈ finRange n := by
  /-
    n : Nat
    a : Fin n
    ⊢ Membership.mem (List.finRange n) a
  -/
  rw [finRange_eq_pmap_range]
  exact mem_pmap.2
    ⟨a.1, mem_range.2 a.2, by
      cases a
      rfl⟩


theorem nodup_finRange (n : ℕ) : (finRange n).Nodup := by
  /-
    n : Nat
    ⊢ (List.finRange n).Nodup
  -/
  rw [finRange_eq_pmap_range]
  /-
    n : Nat
    ⊢ (List.pmap Fin.mk (List.range n) ⋯).Nodup
  -/
  exact (Pairwise.pmap (nodup_range n) _) fun _ _ _ _ => @Fin.ne_of_val_ne _ ⟨_, _⟩ ⟨_, _⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem finRange_eq_nil {n : ℕ} : finRange n = [] ↔ n = 0 := by
  /-
    n : Nat
    ⊢ Iff (Eq (List.finRange n) List.nil) (Eq n 0)
  -/
  rw [← length_eq_zero, length_finRange]
  /-
    🎉 no goals
  -/


theorem pairwise_lt_finRange (n : ℕ) : Pairwise (· < ·) (finRange n) := by
  /-
    n : Nat
    ⊢ List.Pairwise (fun x1 x2 => LT.lt x1 x2) (List.finRange n)
  -/
  rw [finRange_eq_pmap_range]
  /-
    n : Nat
    ⊢ List.Pairwise (fun x1 x2 => LT.lt x1 x2) (List.pmap Fin.mk (List.range n) ⋯)
  -/
  exact (List.pairwise_lt_range n).pmap (by simp) (by simp)
  /-
    🎉 no goals
  -/


theorem pairwise_le_finRange (n : ℕ) : Pairwise (· ≤ ·) (finRange n) := by
  /-
    n : Nat
    ⊢ List.Pairwise (fun x1 x2 => LE.le x1 x2) (List.finRange n)
  -/
  rw [finRange_eq_pmap_range]
  /-
    n : Nat
    ⊢ List.Pairwise (fun x1 x2 => LE.le x1 x2) (List.pmap Fin.mk (List.range n) ⋯)
  -/
  exact (List.pairwise_le_range n).pmap (by simp) (by simp)
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10756): new theorem

theorem get_finRange {n : ℕ} {i : ℕ} (h) :
    (finRange n).get ⟨i, h⟩ = ⟨i, length_finRange n ▸ h⟩ := by
  /-
    n i : Nat
    h : LT.lt i (List.finRange n).length
    ⊢ Eq ((List.finRange n).get ⟨i, h⟩) ⟨i, ⋯⟩
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-19")] alias nthLe_finRange := get_finRange


@[simp]
theorem finRange_map_get (l : List α) : (finRange l.length).map l.get = l :=
                   /-
                     α : Type u
                     l : List α
                     ⊢ Eq (List.map l.get (List.finRange l.length)).length l.length
                   -/
                   /-
                     🎉 no goals
                   -/
  List.ext_get (by simp) (by simp)
                             /-
                               🎉 no goals
                             -/


@[simp] theorem indexOf_finRange {k : ℕ} (i : Fin k) : (finRange k).indexOf i = i := by
  /-
    k : Nat
    i : Fin k
    ⊢ Eq (List.indexOf i (List.finRange k)) ↑i
  -/
  have : (finRange k).indexOf i < (finRange k).length := indexOf_lt_length.mpr (by simp)
  /-
    k : Nat
    i : Fin k
    this : LT.lt (List.indexOf i (List.finRange k)) (List.finRange k).length
    ⊢ Eq (List.indexOf i (List.finRange k)) ↑i
  -/
  have h₁ : (finRange k).get ⟨(finRange k).indexOf i, this⟩ = i := indexOf_get this
  /-
    k : Nat
    i : Fin k
    this : LT.lt (List.indexOf i (List.finRange k)) (List.finRange k).length
    h₁ : Eq ((List.finRange k).get ⟨List.indexOf i (List.finRange k), this⟩) i
    ⊢ Eq (List.indexOf i (List.finRange k)) ↑i
  -/
  have h₂ : (finRange k).get ⟨i, by simp⟩ = i := get_finRange _
  /-
    k : Nat
    i : Fin k
    this : LT.lt (List.indexOf i (List.finRange k)) (List.finRange k).length
    h₁ : Eq ((List.finRange k).get ⟨List.indexOf i (List.finRange k), this⟩) i
    h₂ : Eq ((List.finRange k).get ⟨↑i, ⋯⟩) i
    ⊢ Eq (List.indexOf i (List.finRange k)) ↑i
  -/
  simpa using (Nodup.get_inj_iff (nodup_finRange k)).mp (Eq.trans h₁ h₂.symm)
  /-
    🎉 no goals
  -/


@[simp]
theorem map_coe_finRange (n : ℕ) : ((finRange n) : List (Fin n)).map (Fin.val) = List.range n := by
  /-
    n : Nat
    ⊢ Eq (List.map Fin.val (List.finRange n)) (List.range n)
  -/
                             /-
                               🎉 no goals
                             -/
  apply List.ext_getElem <;> simp
                             /-
                               🎉 no goals
                             -/


theorem finRange_succ_eq_map (n : ℕ) : finRange n.succ = 0 :: (finRange n).map Fin.succ := by
  /-
    n : Nat
    ⊢ Eq (List.finRange n.succ) (List.cons 0 (List.map Fin.succ (List.finRange n)))
  -/
  apply map_injective_iff.mpr Fin.val_injective
  rw [map_cons, map_coe_finRange, range_succ_eq_map, Fin.val_zero, ← map_coe_finRange, map_map,
    map_map]
  /-
    case a
    n : Nat
    ⊢ Eq (List.cons 0 (List.map (Function.comp Nat.succ Fin.val) (List.finRange n) …
  -/
  simp only [Function.comp_def, Fin.val_succ]
  /-
    🎉 no goals
  -/

-- Porting note: `map_nth_le` moved to `List.finRange_map_get` in Data.List.Range


theorem ofFn_eq_pmap {n} {f : Fin n → α} :
    ofFn f = pmap (fun i hi => f ⟨i, hi⟩) (range n) fun _ => mem_range.1 := by
  /-
    α : Type u
    n : Nat
    f : Fin n → α
    ⊢ Eq (List.ofFn f) (List.pmap (fun i hi => f ⟨i, hi⟩) (List.range n) ⋯)
  -/
  rw [pmap_eq_map_attach]
  /-
    α : Type u
    n : Nat
    f : Fin n → α
    ⊢ Eq (List.ofFn f) (List.map (fun x => f ⟨↑x, ⋯⟩) (List.range n).attach)
  -/
  exact ext_getElem (by simp) fun i hi1 hi2 => by simp [List.getElem_ofFn f i hi1]
  /-
    🎉 no goals
  -/


theorem ofFn_id (n) : ofFn id = finRange n :=
  rfl


theorem ofFn_eq_map {n} {f : Fin n → α} : ofFn f = (finRange n).map f := by
  /-
    α : Type u
    n : Nat
    f : Fin n → α
    ⊢ Eq (List.ofFn f) (List.map f (List.finRange n))
  -/
  rw [← ofFn_id, map_ofFn, Function.comp_id]
  /-
    🎉 no goals
  -/


theorem nodup_ofFn_ofInjective {n} {f : Fin n → α} (hf : Function.Injective f) :
    Nodup (ofFn f) := by
  /-
    α : Type u
    n : Nat
    f : Fin n → α
    hf : Function.Injective f
    ⊢ (List.ofFn f).Nodup
  -/
  rw [ofFn_eq_pmap]
  /-
    α : Type u
    n : Nat
    f : Fin n → α
    hf : Function.Injective f
    ⊢ (List.pmap (fun i hi => f ⟨i, hi⟩) (List.range n) ⋯).Nodup
  -/
  exact (nodup_range n).pmap fun _ _ _ _ H => Fin.val_eq_of_eq <| hf H
  /-
    🎉 no goals
  -/


theorem nodup_ofFn {n} {f : Fin n → α} : Nodup (ofFn f) ↔ Function.Injective f := by
  /-
    α : Type u
    n : Nat
    f : Fin n → α
    ⊢ Iff (List.ofFn f).Nodup (Function.Injective f)
  -/
  refine ⟨?_, nodup_ofFn_ofInjective⟩
  /-
    α : Type u
    n : Nat
    f : Fin n → α
    ⊢ (List.ofFn f).Nodup → Function.Injective f
  -/
  refine Fin.consInduction ?_ (fun x₀ xs ih => ?_) f
    /-
      case refine_1
      α : Type u
      n : Nat
      f : Fin n → α
      ⊢ (List.ofFn Fin.elim0).Nodup → Function.Injective Fin.elim0
    -/
  · intro _
    /-
      case refine_1
      α : Type u
      n : Nat
      f : Fin n → α
      a✝ : (List.ofFn Fin.elim0).Nodup
      ⊢ Function.Injective Fin.elim0
    -/
    exact Function.injective_of_subsingleton _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      n : Nat
      f : Fin n → α
      n✝ : Nat
      x₀ : α
      xs : Fin n✝ → α
      ih : (List.ofFn xs).Nodup → Function.Injective xs
      ⊢ (List.ofFn (Fin.cons x₀ xs)).Nodup → Function.Injective (Fin.cons x₀ xs)
    -/
  · intro h
    /-
      case refine_2
      α : Type u
      n : Nat
      f : Fin n → α
      n✝ : Nat
      x₀ : α
      xs : Fin n✝ → α
      ih : (List.ofFn xs).Nodup → Function.Injective xs
      h : (List.ofFn (Fin.cons x₀ xs)).Nodup
      ⊢ Function.Injective (Fin.cons x₀ xs)
    -/
    rw [Fin.cons_injective_iff]
    /-
      case refine_2
      α : Type u
      n : Nat
      f : Fin n → α
      n✝ : Nat
      x₀ : α
      xs : Fin n✝ → α
      ih : (List.ofFn xs).Nodup → Function.Injective xs
      h : (List.ofFn (Fin.cons x₀ xs)).Nodup
      ⊢ And (Not (Membership.mem (Set.range xs) x₀)) (Function.Injective xs)
    -/
    simp_rw [ofFn_succ, Fin.cons_succ, nodup_cons, Fin.cons_zero, mem_ofFn] at h
    /-
      case refine_2
      α : Type u
      n : Nat
      f : Fin n → α
      n✝ : Nat
      x₀ : α
      xs : Fin n✝ → α
      ih : (List.ofFn xs).Nodup → Function.Injective xs
      h : And (Not (Membership.mem (Set.range fun i => xs i) x₀)) (List.ofFn fun i = …
      ⊢ And (Not (Membership.mem (Set.range xs) x₀)) (Function.Injective xs)
    -/
    exact h.imp_right ih
    /-
      🎉 no goals
    -/


theorem Equiv.Perm.map_finRange_perm {n : ℕ} (σ : Equiv.Perm (Fin n)) :
    map σ (finRange n) ~ finRange n := by
  /-
    n : Nat
    σ : Equiv.Perm (Fin n)
    ⊢ (List.map (⇑σ) (List.finRange n)).Perm (List.finRange n)
  -/
  rw [perm_ext_iff_of_nodup ((nodup_finRange n).map σ.injective) <| nodup_finRange n]
  /-
    n : Nat
    σ : Equiv.Perm (Fin n)
    ⊢ ∀ (a : Fin n), Iff (Membership.mem (List.map (⇑σ) (List.finRange n)) a) (Mem …
  -/
  simpa [mem_map, mem_finRange] using σ.surjective
  /-
    🎉 no goals
  -/


/-- The list obtained from a permutation of a tuple `f` is permutation equivalent to
the list obtained from `f`. -/
theorem Equiv.Perm.ofFn_comp_perm {n : ℕ} {α : Type u} (σ : Equiv.Perm (Fin n)) (f : Fin n → α) :
    ofFn (f ∘ σ) ~ ofFn f := by
  /-
    n : Nat
    α : Type u
    σ : Equiv.Perm (Fin n)
    f : Fin n → α
    ⊢ (List.ofFn (Function.comp f ⇑σ)).Perm (List.ofFn f)
  -/
  rw [ofFn_eq_map, ofFn_eq_map, ← map_map]
  /-
    n : Nat
    α : Type u
    σ : Equiv.Perm (Fin n)
    f : Fin n → α
    ⊢ (List.map f (List.map (⇑σ) (List.finRange n))).Perm (List.map f (List.finRan …
  -/
  exact σ.map_finRange_perm.map f
  /-
    🎉 no goals
  -/

