/-- If `f : ℕ → α` is a sequence of elements, then `disjointed f` is the sequence formed by
subtracting each element from the nexts. This is the unique disjoint sequence whose partial sups
are the same as the original sequence. -/
def disjointed (f : ℕ → α) : ℕ → α
  | 0 => f 0
  | n + 1 => f (n + 1) \ partialSups f n


@[simp]
theorem disjointed_zero (f : ℕ → α) : disjointed f 0 = f 0 :=
  rfl


theorem disjointed_succ (f : ℕ → α) (n : ℕ) : disjointed f (n + 1) = f (n + 1) \ partialSups f n :=
  rfl


theorem disjointed_le_id : disjointed ≤ (id : (ℕ → α) → ℕ → α) := by
  /-
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    ⊢ LE.le disjointed id
  -/
  rintro f n
  /-
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f : Nat → α
    n : Nat
    ⊢ LE.le (disjointed f n) (id f n)
  -/
  cases n
    /-
      case zero
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f : Nat → α
      ⊢ LE.le (disjointed f 0) (id f 0)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f : Nat → α
      n✝ : Nat
      ⊢ LE.le (disjointed f (HAdd.hAdd n✝ 1)) (id f (HAdd.hAdd n✝ 1))
    -/
  · exact sdiff_le
    /-
      🎉 no goals
    -/


theorem disjointed_le (f : ℕ → α) : disjointed f ≤ f :=
  disjointed_le_id f


theorem disjoint_disjointed (f : ℕ → α) : Pairwise (Disjoint on disjointed f) := by
  /-
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f : Nat → α
    ⊢ Pairwise (Function.onFun Disjoint (disjointed f))
  -/
  refine (Symmetric.pairwise_on Disjoint.symm _).2 fun m n h => ?_
  /-
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f : Nat → α
    m n : Nat
    h : LT.lt m n
    ⊢ Disjoint (disjointed f m) (disjointed f n)
  -/
  cases n
    /-
      case zero
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f : Nat → α
      m : Nat
      h : LT.lt m 0
      ⊢ Disjoint (disjointed f m) (disjointed f 0)
    -/
  · exact (Nat.not_lt_zero _ h).elim
    /-
      🎉 no goals
    -/
  exact
    disjoint_sdiff_self_right.mono_left
      ((disjointed_le f m).trans (le_partialSups_of_le f (Nat.lt_add_one_iff.1 h)))

-- Porting note: `disjointedRec` had a change in universe level.

/-- An induction principle for `disjointed`. To define/prove something on `disjointed f n`, it's
enough to define/prove it for `f n` and being able to extend through diffs. -/
def disjointedRec {f : ℕ → α} {p : α → Sort*} (hdiff : ∀ ⦃t i⦄, p t → p (t \ f i)) :
    ∀ ⦃n⦄, p (f n) → p (disjointed f n)
  | 0 => id
  | n + 1 => fun h => by
    /-
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f : Nat → α
      p : α → Sort u_2
      hdiff : ⦃t : α⦄ → ⦃i : Nat⦄ → p t → p (SDiff.sdiff t (f i))
      n : Nat
      h : p (f (HAdd.hAdd n 1))
      ⊢ p (disjointed f (HAdd.hAdd n 1))
    -/
    suffices H : ∀ k, p (f (n + 1) \ partialSups f k) from H n
    /-
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f : Nat → α
      p : α → Sort u_2
      hdiff : ⦃t : α⦄ → ⦃i : Nat⦄ → p t → p (SDiff.sdiff t (f i))
      n : Nat
      h : p (f (HAdd.hAdd n 1))
      ⊢ (k : Nat) → p (SDiff.sdiff (f (HAdd.hAdd n 1)) ((partialSups f) k))
    -/
    rintro k
    /-
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f : Nat → α
      p : α → Sort u_2
      hdiff : ⦃t : α⦄ → ⦃i : Nat⦄ → p t → p (SDiff.sdiff t (f i))
      n : Nat
      h : p (f (HAdd.hAdd n 1))
      k : Nat
      ⊢ p (SDiff.sdiff (f (HAdd.hAdd n 1)) ((partialSups f) k))
    -/
    induction' k with k ih
      /-
        case zero
        α : Type u_1
        inst✝ : GeneralizedBooleanAlgebra α
        f : Nat → α
        p : α → Sort u_2
        hdiff : ⦃t : α⦄ → ⦃i : Nat⦄ → p t → p (SDiff.sdiff t (f i))
        n : Nat
        h : p (f (HAdd.hAdd n 1))
        ⊢ p (SDiff.sdiff (f (HAdd.hAdd n 1)) ((partialSups f) 0))
      -/
    · exact hdiff h
      /-
        🎉 no goals
      -/
    /-
      case succ
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f : Nat → α
      p : α → Sort u_2
      hdiff : ⦃t : α⦄ → ⦃i : Nat⦄ → p t → p (SDiff.sdiff t (f i))
      n : Nat
      h : p (f (HAdd.hAdd n 1))
      k : Nat
      ih : p (SDiff.sdiff (f (HAdd.hAdd n 1)) ((partialSups f) k))
      ⊢ p (SDiff.sdiff (f (HAdd.hAdd n 1)) ((partialSups f) (HAdd.hAdd k 1)))
    -/
    rw [partialSups_succ, ← sdiff_sdiff_left]
    /-
      case succ
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f : Nat → α
      p : α → Sort u_2
      hdiff : ⦃t : α⦄ → ⦃i : Nat⦄ → p t → p (SDiff.sdiff t (f i))
      n : Nat
      h : p (f (HAdd.hAdd n 1))
      k : Nat
      ih : p (SDiff.sdiff (f (HAdd.hAdd n 1)) ((partialSups f) k))
      ⊢ p (SDiff.sdiff (SDiff.sdiff (f (HAdd.hAdd n 1)) ((partialSups f) k)) (f (HAd …
    -/
    exact hdiff ih
    /-
      🎉 no goals
    -/


@[simp]
theorem disjointedRec_zero {f : ℕ → α} {p : α → Sort*} (hdiff : ∀ ⦃t i⦄, p t → p (t \ f i))
    (h₀ : p (f 0)) : disjointedRec hdiff h₀ = h₀ :=
  rfl

-- TODO: Find a useful statement of `disjointedRec_succ`.

protected lemma Monotone.disjointed_succ {f : ℕ → α} (hf : Monotone f) (n : ℕ) :
                                                 /-
                                                   α : Type u_1
                                                   inst✝ : GeneralizedBooleanAlgebra α
                                                   f : Nat → α
                                                   hf : Monotone f
                                                   n : Nat
                                                   ⊢ Eq (disjointed f (HAdd.hAdd n 1)) (SDiff.sdiff (f (HAdd.hAdd n 1)) (f n))
                                                 -/
    disjointed f (n + 1) = f (n + 1) \ f n := by rw [disjointed_succ, hf.partialSups_eq]
                                                 /-
                                                   🎉 no goals
                                                 -/


protected lemma Monotone.disjointed_succ_sup {f : ℕ → α} (hf : Monotone f) (n : ℕ) :
    disjointed f (n + 1) ⊔ f n = f (n + 1) := by
  /-
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f : Nat → α
    hf : Monotone f
    n : Nat
    ⊢ Eq (Max.max (disjointed f (HAdd.hAdd n 1)) (f n)) (f (HAdd.hAdd n 1))
  -/
  rw [hf.disjointed_succ, sdiff_sup_cancel]; exact hf n.le_succ
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem partialSups_disjointed (f : ℕ → α) : partialSups (disjointed f) = partialSups f := by
  /-
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f : Nat → α
    ⊢ Eq (partialSups (disjointed f)) (partialSups f)
  -/
  ext n
  /-
    case h.h
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f : Nat → α
    n : Nat
    ⊢ Eq ((partialSups (disjointed f)) n) ((partialSups f) n)
  -/
  induction' n with k ih
    /-
      case h.h.zero
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f : Nat → α
      ⊢ Eq ((partialSups (disjointed f)) 0) ((partialSups f) 0)
    -/
  · rw [partialSups_zero, partialSups_zero, disjointed_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.h.succ
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f : Nat → α
      k : Nat
      ih : Eq ((partialSups (disjointed f)) k) ((partialSups f) k)
      ⊢ Eq ((partialSups (disjointed f)) (HAdd.hAdd k 1)) ((partialSups f) (HAdd.hAd …
    -/
  · rw [partialSups_succ, partialSups_succ, disjointed_succ, ih, sup_sdiff_self_right]
    /-
      🎉 no goals
    -/


/-- `disjointed f` is the unique sequence that is pairwise disjoint and has the same partial sups
as `f`. -/
theorem disjointed_unique {f d : ℕ → α} (hdisj : Pairwise (Disjoint on d))
    (hsups : partialSups d = partialSups f) : d = disjointed f := by
  /-
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f d : Nat → α
    hdisj : Pairwise (Function.onFun Disjoint d)
    hsups : Eq (partialSups d) (partialSups f)
    ⊢ Eq d (disjointed f)
  -/
  ext n
  /-
    case h
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f d : Nat → α
    hdisj : Pairwise (Function.onFun Disjoint d)
    hsups : Eq (partialSups d) (partialSups f)
    n : Nat
    ⊢ Eq (d n) (disjointed f n)
  -/
  cases' n with n
    /-
      case h.zero
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f d : Nat → α
      hdisj : Pairwise (Function.onFun Disjoint d)
      hsups : Eq (partialSups d) (partialSups f)
      ⊢ Eq (d 0) (disjointed f 0)
    -/
  · rw [← partialSups_zero d, hsups, partialSups_zero, disjointed_zero]
    /-
      🎉 no goals
    -/
  suffices h : d n.succ = partialSups d n.succ \ partialSups d n by
    rw [h, hsups, partialSups_succ, disjointed_succ, sup_sdiff, sdiff_self, bot_sup_eq]
  /-
    case h.succ
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f d : Nat → α
    hdisj : Pairwise (Function.onFun Disjoint d)
    hsups : Eq (partialSups d) (partialSups f)
    n : Nat
    ⊢ Eq (d n.succ) (SDiff.sdiff ((partialSups d) n.succ) ((partialSups d) n))
  -/
  rw [partialSups_succ, sup_sdiff, sdiff_self, bot_sup_eq, eq_comm, sdiff_eq_self_iff_disjoint]
  /-
    case h.succ
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f d : Nat → α
    hdisj : Pairwise (Function.onFun Disjoint d)
    hsups : Eq (partialSups d) (partialSups f)
    n : Nat
    ⊢ Disjoint ((partialSups d) n) (d (HAdd.hAdd n 1))
  -/
  suffices h : ∀ m ≤ n, Disjoint (partialSups d m) (d n.succ) from h n le_rfl
  /-
    case h.succ
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f d : Nat → α
    hdisj : Pairwise (Function.onFun Disjoint d)
    hsups : Eq (partialSups d) (partialSups f)
    n : Nat
    ⊢ ∀ (m : Nat), LE.le m n → Disjoint ((partialSups d) m) (d n.succ)
  -/
  rintro m hm
  /-
    case h.succ
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f d : Nat → α
    hdisj : Pairwise (Function.onFun Disjoint d)
    hsups : Eq (partialSups d) (partialSups f)
    n m : Nat
    hm : LE.le m n
    ⊢ Disjoint ((partialSups d) m) (d n.succ)
  -/
  induction' m with m ih
    /-
      case h.succ.zero
      α : Type u_1
      inst✝ : GeneralizedBooleanAlgebra α
      f d : Nat → α
      hdisj : Pairwise (Function.onFun Disjoint d)
      hsups : Eq (partialSups d) (partialSups f)
      n : Nat
      hm : LE.le 0 n
      ⊢ Disjoint ((partialSups d) 0) (d n.succ)
    -/
  · exact hdisj (Nat.succ_ne_zero _).symm
    /-
      🎉 no goals
    -/
  /-
    case h.succ.succ
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f d : Nat → α
    hdisj : Pairwise (Function.onFun Disjoint d)
    hsups : Eq (partialSups d) (partialSups f)
    n m : Nat
    ih : LE.le m n → Disjoint ((partialSups d) m) (d n.succ)
    hm : LE.le (HAdd.hAdd m 1) n
    ⊢ Disjoint ((partialSups d) (HAdd.hAdd m 1)) (d n.succ)
  -/
  rw [partialSups_succ, disjoint_iff, inf_sup_right, sup_eq_bot_iff, ← disjoint_iff, ← disjoint_iff]
  /-
    case h.succ.succ
    α : Type u_1
    inst✝ : GeneralizedBooleanAlgebra α
    f d : Nat → α
    hdisj : Pairwise (Function.onFun Disjoint d)
    hsups : Eq (partialSups d) (partialSups f)
    n m : Nat
    ih : LE.le m n → Disjoint ((partialSups d) m) (d n.succ)
    hm : LE.le (HAdd.hAdd m 1) n
    ⊢ And (Disjoint ((partialSups d) m) (d n.succ)) (Disjoint (d (HAdd.hAdd m 1))  …
  -/
  exact ⟨ih (Nat.le_of_succ_le hm), hdisj (Nat.lt_succ_of_le hm).ne⟩
  /-
    🎉 no goals
  -/


theorem iSup_disjointed (f : ℕ → α) : ⨆ n, disjointed f n = ⨆ n, f n :=
  iSup_eq_iSup_of_partialSups_eq_partialSups (partialSups_disjointed f)


theorem disjointed_eq_inf_compl (f : ℕ → α) (n : ℕ) : disjointed f n = f n ⊓ ⨅ i < n, (f i)ᶜ := by
  /-
    α : Type u_1
    inst✝ : CompleteBooleanAlgebra α
    f : Nat → α
    n : Nat
    ⊢ Eq (disjointed f n) (Min.min (f n) (iInf fun i => iInf fun h => HasCompl.com …
  -/
  cases n
    /-
      case zero
      α : Type u_1
      inst✝ : CompleteBooleanAlgebra α
      f : Nat → α
      ⊢ Eq (disjointed f 0) (Min.min (f 0) (iInf fun i => iInf fun h => HasCompl.com …
    -/
  · rw [disjointed_zero, eq_comm, inf_eq_left]
    /-
      case zero
      α : Type u_1
      inst✝ : CompleteBooleanAlgebra α
      f : Nat → α
      ⊢ LE.le (f 0) (iInf fun i => iInf fun h => HasCompl.compl (f i))
    -/
    simp_rw [le_iInf_iff]
    /-
      case zero
      α : Type u_1
      inst✝ : CompleteBooleanAlgebra α
      f : Nat → α
      ⊢ ∀ (i : Nat), LT.lt i 0 → LE.le (f 0) (HasCompl.compl (f i))
    -/
    exact fun i hi => (i.not_lt_zero hi).elim
    /-
      🎉 no goals
    -/
  /-
    case succ
    α : Type u_1
    inst✝ : CompleteBooleanAlgebra α
    f : Nat → α
    n✝ : Nat
    ⊢ Eq (disjointed f (HAdd.hAdd n✝ 1)) (Min.min (f (HAdd.hAdd n✝ 1)) (iInf fun i …
  -/
  simp_rw [disjointed_succ, partialSups_eq_biSup, sdiff_eq, compl_iSup]
  /-
    case succ
    α : Type u_1
    inst✝ : CompleteBooleanAlgebra α
    f : Nat → α
    n✝ : Nat
    ⊢ Eq (Min.min (f (HAdd.hAdd n✝ 1)) (iInf fun i => iInf fun i_1 => HasCompl.com …
  -/
  congr
  /-
    case succ.e_a.e_s
    α : Type u_1
    inst✝ : CompleteBooleanAlgebra α
    f : Nat → α
    n✝ : Nat
    ⊢ Eq (fun i => iInf fun i_1 => HasCompl.compl (f i)) fun i => iInf fun h => Ha …
  -/
  ext i
  /-
    case succ.e_a.e_s.h
    α : Type u_1
    inst✝ : CompleteBooleanAlgebra α
    f : Nat → α
    n✝ i : Nat
    ⊢ Eq (iInf fun i_1 => HasCompl.compl (f i)) (iInf fun h => HasCompl.compl (f i))
  -/
  rw [Nat.lt_succ_iff]
  /-
    🎉 no goals
  -/


theorem disjointed_subset (f : ℕ → Set α) (n : ℕ) : disjointed f n ⊆ f n :=
  disjointed_le f n


theorem iUnion_disjointed {f : ℕ → Set α} : ⋃ n, disjointed f n = ⋃ n, f n :=
  iSup_disjointed f


theorem disjointed_eq_inter_compl (f : ℕ → Set α) (n : ℕ) :
    disjointed f n = f n ∩ ⋂ i < n, (f i)ᶜ :=
  disjointed_eq_inf_compl f n


theorem preimage_find_eq_disjointed (s : ℕ → Set α) (H : ∀ x, ∃ n, x ∈ s n)
    [∀ x n, Decidable (x ∈ s n)] (n : ℕ) : (fun x => Nat.find (H x)) ⁻¹' {n} = disjointed s n := by
  /-
    α : Type u_1
    s : Nat → Set α
    H : ∀ (x : α), Exists fun n => Membership.mem (s n) x
    inst✝ : (x : α) → (n : Nat) → Decidable (Membership.mem (s n) x)
    n : Nat
    ⊢ Eq (Set.preimage (fun x => Nat.find ⋯) (Singleton.singleton n)) (disjointed  …
  -/
  ext x
  /-
    case h
    α : Type u_1
    s : Nat → Set α
    H : ∀ (x : α), Exists fun n => Membership.mem (s n) x
    inst✝ : (x : α) → (n : Nat) → Decidable (Membership.mem (s n) x)
    n : Nat
    x : α
    ⊢ Iff (Membership.mem (Set.preimage (fun x => Nat.find ⋯) (Singleton.singleton …
  -/
  simp [Nat.find_eq_iff, disjointed_eq_inter_compl]
  /-
    🎉 no goals
  -/

