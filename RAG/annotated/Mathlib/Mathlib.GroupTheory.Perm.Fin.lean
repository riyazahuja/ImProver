/-- Permutations of `Fin (n + 1)` are equivalent to fixing a single
`Fin (n + 1)` and permuting the remaining with a `Perm (Fin n)`.
The fixed `Fin (n + 1)` is swapped with `0`. -/
def Equiv.Perm.decomposeFin {n : ℕ} : Perm (Fin n.succ) ≃ Fin n.succ × Perm (Fin n) :=
  ((Equiv.permCongr <| finSuccEquiv n).trans Equiv.Perm.decomposeOption).trans
    (Equiv.prodCongr (finSuccEquiv n).symm (Equiv.refl _))


@[simp]
theorem Equiv.Perm.decomposeFin_symm_of_refl {n : ℕ} (p : Fin (n + 1)) :
    Equiv.Perm.decomposeFin.symm (p, Equiv.refl _) = swap 0 p := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Equiv.Perm.decomposeFin.symm { fst := p, snd := Equiv.refl (Fin n) }) (E …
  -/
  simp [Equiv.Perm.decomposeFin, Equiv.permCongr_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem Equiv.Perm.decomposeFin_symm_of_one {n : ℕ} (p : Fin (n + 1)) :
    Equiv.Perm.decomposeFin.symm (p, 1) = swap 0 p :=
  Equiv.Perm.decomposeFin_symm_of_refl p


@[simp]
theorem Equiv.Perm.decomposeFin_symm_apply_zero {n : ℕ} (p : Fin (n + 1)) (e : Perm (Fin n)) :
                                                    /-
                                                      n : Nat
                                                      p : Fin (HAdd.hAdd n 1)
                                                      e : Equiv.Perm (Fin n)
                                                      ⊢ Eq ((Equiv.Perm.decomposeFin.symm { fst := p, snd := e }) 0) p
                                                    -/
    Equiv.Perm.decomposeFin.symm (p, e) 0 = p := by simp [Equiv.Perm.decomposeFin]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem Equiv.Perm.decomposeFin_symm_apply_succ {n : ℕ} (e : Perm (Fin n)) (p : Fin (n + 1))
    (x : Fin n) : Equiv.Perm.decomposeFin.symm (p, e) x.succ = swap 0 p (e x).succ := by
  /-
    n : Nat
    e : Equiv.Perm (Fin n)
    p : Fin (HAdd.hAdd n 1)
    x : Fin n
    ⊢ Eq ((Equiv.Perm.decomposeFin.symm { fst := p, snd := e }) x.succ) ((Equiv.sw …
  -/
  refine Fin.cases ?_ ?_ p
    /-
      case refine_1
      n : Nat
      e : Equiv.Perm (Fin n)
      p : Fin (HAdd.hAdd n 1)
      x : Fin n
      ⊢ Eq ((Equiv.Perm.decomposeFin.symm { fst := 0, snd := e }) x.succ) ((Equiv.sw …
    -/
  · simp [Equiv.Perm.decomposeFin, EquivFunctor.map]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      e : Equiv.Perm (Fin n)
      p : Fin (HAdd.hAdd n 1)
      x : Fin n
      ⊢ ∀ (i : Fin n), Eq ((Equiv.Perm.decomposeFin.symm { fst := i.succ, snd := e } …
    -/
  · intro i
    /-
      case refine_2
      n : Nat
      e : Equiv.Perm (Fin n)
      p : Fin (HAdd.hAdd n 1)
      x i : Fin n
      ⊢ Eq ((Equiv.Perm.decomposeFin.symm { fst := i.succ, snd := e }) x.succ) ((Equ …
    -/
    by_cases h : i = e x
      /-
        case pos
        n : Nat
        e : Equiv.Perm (Fin n)
        p : Fin (HAdd.hAdd n 1)
        x i : Fin n
        h : Eq i (e x)
        ⊢ Eq ((Equiv.Perm.decomposeFin.symm { fst := i.succ, snd := e }) x.succ) ((Equ …
      -/
    · simp [h, Equiv.Perm.decomposeFin, EquivFunctor.map]
      /-
        🎉 no goals
      -/
    · simp [h, Fin.succ_ne_zero, Equiv.Perm.decomposeFin, EquivFunctor.map,
        swap_apply_def, Ne.symm h]


@[simp]
theorem Equiv.Perm.decomposeFin_symm_apply_one {n : ℕ} (e : Perm (Fin (n + 1))) (p : Fin (n + 2)) :
    Equiv.Perm.decomposeFin.symm (p, e) 1 = swap 0 p (e 0).succ := by
  /-
    n : Nat
    e : Equiv.Perm (Fin (HAdd.hAdd n 1))
    p : Fin (HAdd.hAdd n 2)
    ⊢ Eq ((Equiv.Perm.decomposeFin.symm { fst := p, snd := e }) 1) ((Equiv.swap 0  …
  -/
  rw [← Fin.succ_zero_eq_one, Equiv.Perm.decomposeFin_symm_apply_succ e p 0]
  /-
    🎉 no goals
  -/


@[simp]
theorem Equiv.Perm.decomposeFin.symm_sign {n : ℕ} (p : Fin (n + 1)) (e : Perm (Fin n)) :
    Perm.sign (Equiv.Perm.decomposeFin.symm (p, e)) = ite (p = 0) 1 (-1) * Perm.sign e := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    e : Equiv.Perm (Fin n)
    ⊢ Eq (Equiv.Perm.sign (Equiv.Perm.decomposeFin.symm { fst := p, snd := e })) ( …
  -/
                               /-
                                 🎉 no goals
                               -/
  refine Fin.cases ?_ ?_ p <;> simp [Equiv.Perm.decomposeFin, Fin.succ_ne_zero]
                               /-
                                 🎉 no goals
                               -/


/-- The set of all permutations of `Fin (n + 1)` can be constructed by augmenting the set of
permutations of `Fin n` by each element of `Fin (n + 1)` in turn. -/
theorem Finset.univ_perm_fin_succ {n : ℕ} :
    @Finset.univ (Perm <| Fin n.succ) _ =
      (Finset.univ : Finset <| Fin n.succ × Perm (Fin n)).map
        Equiv.Perm.decomposeFin.symm.toEmbedding :=
  (Finset.univ_map_equiv_to_embedding _).symm


theorem finRotate_succ_eq_decomposeFin {n : ℕ} :
    finRotate n.succ = decomposeFin.symm (1, finRotate n) := by
  /-
    n : Nat
    ⊢ Eq (finRotate n.succ) (Equiv.Perm.decomposeFin.symm { fst := 1, snd := finRo …
  -/
  ext i
  /-
    case H.h
    n : Nat
    i : Fin n.succ
    ⊢ Eq ↑((finRotate n.succ) i) ↑((Equiv.Perm.decomposeFin.symm { fst := 1, snd : …
  -/
  cases n; · simp
             /-
               🎉 no goals
             -/
  /-
    case H.h.succ
    n✝ : Nat
    i : Fin (HAdd.hAdd n✝ 1).succ
    ⊢ Eq ↑((finRotate (HAdd.hAdd n✝ 1).succ) i) ↑((Equiv.Perm.decomposeFin.symm {  …
  -/
  refine Fin.cases ?_ (fun i => ?_) i
    /-
      case H.h.succ.refine_1
      n✝ : Nat
      i : Fin (HAdd.hAdd n✝ 1).succ
      ⊢ Eq ↑((finRotate (HAdd.hAdd n✝ 1).succ) 0) ↑((Equiv.Perm.decomposeFin.symm {  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case H.h.succ.refine_2
    n✝ : Nat
    i✝ : Fin (HAdd.hAdd n✝ 1).succ
    i : Fin (HAdd.hAdd n✝ 1)
    ⊢ Eq ↑((finRotate (HAdd.hAdd n✝ 1).succ) i.succ) ↑((Equiv.Perm.decomposeFin.sy …
  -/
  rw [coe_finRotate, decomposeFin_symm_apply_succ, if_congr i.succ_eq_last_succ rfl rfl]
  /-
    case H.h.succ.refine_2
    n✝ : Nat
    i✝ : Fin (HAdd.hAdd n✝ 1).succ
    i : Fin (HAdd.hAdd n✝ 1)
    ⊢ Eq (ite (Eq i (Fin.last n✝)) 0 (HAdd.hAdd (↑i.succ) 1)) ↑((Equiv.swap 0 1) ( …
  -/
  split_ifs with h
    /-
      case pos
      n✝ : Nat
      i✝ : Fin (HAdd.hAdd n✝ 1).succ
      i : Fin (HAdd.hAdd n✝ 1)
      h : Eq i (Fin.last n✝)
      ⊢ Eq 0 ↑((Equiv.swap 0 1) ((finRotate (HAdd.hAdd n✝ 1)) i).succ)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  · rw [Fin.val_succ, Function.Injective.map_swap Fin.val_injective, Fin.val_succ, coe_finRotate,
      if_neg h, Fin.val_zero, Fin.val_one,
      swap_apply_of_ne_of_ne (Nat.succ_ne_zero _) (Nat.succ_succ_ne_one _)]


@[simp]
theorem sign_finRotate (n : ℕ) : Perm.sign (finRotate (n + 1)) = (-1) ^ n := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [finRotate_succ_eq_decomposeFin]
    simp [ih, pow_succ]


@[simp]
theorem support_finRotate {n : ℕ} : support (finRotate (n + 2)) = Finset.univ := by
  /-
    n : Nat
    ⊢ Eq (finRotate (HAdd.hAdd n 2)).support Finset.univ
  -/
  ext
  /-
    case h
    n : Nat
    a✝ : Fin (HAdd.hAdd n 2)
    ⊢ Iff (Membership.mem (finRotate (HAdd.hAdd n 2)).support a✝) (Membership.mem  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem support_finRotate_of_le {n : ℕ} (h : 2 ≤ n) : support (finRotate n) = Finset.univ := by
  /-
    n : Nat
    h : LE.le 2 n
    ⊢ Eq (finRotate n).support Finset.univ
  -/
  obtain ⟨m, rfl⟩ := exists_add_of_le h
  /-
    case intro
    m : Nat
    h : LE.le 2 (HAdd.hAdd 2 m)
    ⊢ Eq (finRotate (HAdd.hAdd 2 m)).support Finset.univ
  -/
  rw [add_comm, support_finRotate]
  /-
    🎉 no goals
  -/


theorem isCycle_finRotate {n : ℕ} : IsCycle (finRotate (n + 2)) := by
  /-
    n : Nat
    ⊢ (finRotate (HAdd.hAdd n 2)).IsCycle
  -/
  refine ⟨0, by simp, fun x hx' => ⟨x, ?_⟩⟩
  /-
    n : Nat
    x : Fin (HAdd.hAdd n 2)
    hx' : Ne ((finRotate (HAdd.hAdd n 2)) x) x
    ⊢ Eq ((HPow.hPow (finRotate (HAdd.hAdd n 2)) ↑↑x) 0) x
  -/
  clear hx'
  /-
    n : Nat
    x : Fin (HAdd.hAdd n 2)
    ⊢ Eq ((HPow.hPow (finRotate (HAdd.hAdd n 2)) ↑↑x) 0) x
  -/
  cases' x with x hx
  /-
    case mk
    n x : Nat
    hx : LT.lt x (HAdd.hAdd n 2)
    ⊢ Eq ((HPow.hPow (finRotate (HAdd.hAdd n 2)) ↑↑⟨x, hx⟩) 0) ⟨x, hx⟩
  -/
  rw [zpow_natCast, Fin.ext_iff, Fin.val_mk]
  /-
    case mk
    n x : Nat
    hx : LT.lt x (HAdd.hAdd n 2)
    ⊢ Eq ↑((HPow.hPow (finRotate (HAdd.hAdd n 2)) ↑⟨x, hx⟩) 0) ↑⟨x, hx⟩
  -/
  induction' x with x ih; · rfl
                            /-
                              🎉 no goals
                            -/
  /-
    case mk.succ
    n x : Nat
    ih : ∀ (hx : LT.lt x (HAdd.hAdd n 2)), Eq ↑((HPow.hPow (finRotate (HAdd.hAdd n …
    hx : LT.lt (HAdd.hAdd x 1) (HAdd.hAdd n 2)
    ⊢ Eq ↑((HPow.hPow (finRotate (HAdd.hAdd n 2)) ↑⟨HAdd.hAdd x 1, hx⟩) 0) ↑⟨HAdd. …
  -/
  rw [pow_succ', Perm.mul_apply, coe_finRotate_of_ne_last, ih (lt_trans x.lt_succ_self hx)]
  /-
    case mk.succ
    n x : Nat
    ih : ∀ (hx : LT.lt x (HAdd.hAdd n 2)), Eq ↑((HPow.hPow (finRotate (HAdd.hAdd n …
    hx : LT.lt (HAdd.hAdd x 1) (HAdd.hAdd n 2)
    ⊢ Ne ((HPow.hPow (finRotate (HAdd.hAdd n 2)) x) 0) (Fin.last (HAdd.hAdd n 1))
  -/
  rw [Ne, Fin.ext_iff, ih (lt_trans x.lt_succ_self hx), Fin.val_last]
  /-
    case mk.succ
    n x : Nat
    ih : ∀ (hx : LT.lt x (HAdd.hAdd n 2)), Eq ↑((HPow.hPow (finRotate (HAdd.hAdd n …
    hx : LT.lt (HAdd.hAdd x 1) (HAdd.hAdd n 2)
    ⊢ Not (Eq (↑⟨x, ⋯⟩) (HAdd.hAdd n 1))
  -/
  exact ne_of_lt (Nat.lt_of_succ_lt_succ hx)
  /-
    🎉 no goals
  -/


theorem isCycle_finRotate_of_le {n : ℕ} (h : 2 ≤ n) : IsCycle (finRotate n) := by
  /-
    n : Nat
    h : LE.le 2 n
    ⊢ (finRotate n).IsCycle
  -/
  obtain ⟨m, rfl⟩ := exists_add_of_le h
  /-
    case intro
    m : Nat
    h : LE.le 2 (HAdd.hAdd 2 m)
    ⊢ (finRotate (HAdd.hAdd 2 m)).IsCycle
  -/
  rw [add_comm]
  /-
    case intro
    m : Nat
    h : LE.le 2 (HAdd.hAdd 2 m)
    ⊢ (finRotate (HAdd.hAdd m 2)).IsCycle
  -/
  exact isCycle_finRotate
  /-
    🎉 no goals
  -/


@[simp]
theorem cycleType_finRotate {n : ℕ} : cycleType (finRotate (n + 2)) = {n + 2} := by
  /-
    n : Nat
    ⊢ Eq (finRotate (HAdd.hAdd n 2)).cycleType (Singleton.singleton (HAdd.hAdd n 2))
  -/
  rw [isCycle_finRotate.cycleType, support_finRotate, ← Fintype.card, Fintype.card_fin]
  /-
    n : Nat
    ⊢ Eq (↑(List.cons (HAdd.hAdd n 2) List.nil)) (Singleton.singleton (HAdd.hAdd n …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem cycleType_finRotate_of_le {n : ℕ} (h : 2 ≤ n) : cycleType (finRotate n) = {n} := by
  /-
    n : Nat
    h : LE.le 2 n
    ⊢ Eq (finRotate n).cycleType (Singleton.singleton n)
  -/
  obtain ⟨m, rfl⟩ := exists_add_of_le h
  /-
    case intro
    m : Nat
    h : LE.le 2 (HAdd.hAdd 2 m)
    ⊢ Eq (finRotate (HAdd.hAdd 2 m)).cycleType (Singleton.singleton (HAdd.hAdd 2 m))
  -/
  rw [add_comm, cycleType_finRotate]
  /-
    🎉 no goals
  -/


/-- `Fin.cycleRange i` is the cycle `(0 1 2 ... i)` leaving `(i+1 ... (n-1))` unchanged. -/
def cycleRange {n : ℕ} (i : Fin n) : Perm (Fin n) :=
  (finRotate (i + 1)).extendDomain
    (Equiv.ofLeftInverse' (Fin.castLEEmb (Nat.succ_le_of_lt i.is_lt)) (↑)
      (by
        /-
          n : Nat
          i : Fin n
          ⊢ Function.LeftInverse (fun x => ↑↑x) ⇑(Fin.castLEEmb ⋯)
        -/
        intro x
        /-
          n : Nat
          i : Fin n
          x : Fin (↑i).succ
          ⊢ Eq ((fun x => ↑↑x) ((Fin.castLEEmb ⋯) x)) x
        -/
        ext
        /-
          case h
          n : Nat
          i : Fin n
          x : Fin (↑i).succ
          ⊢ Eq ↑((fun x => ↑↑x) ((Fin.castLEEmb ⋯) x)) ↑x
        -/
        simp))
        /-
          🎉 no goals
        -/


theorem cycleRange_of_gt {n : ℕ} {i j : Fin n.succ} (h : i < j) : cycleRange i j = j := by
  rw [cycleRange, ofLeftInverse'_eq_ofInjective,
    ← Function.Embedding.toEquivRange_eq_ofInjective, ← viaFintypeEmbedding,
    viaFintypeEmbedding_apply_not_mem_range]
  /-
    case h
    n : Nat
    i j : Fin n.succ
    h : LT.lt i j
    ⊢ Not (Membership.mem (Set.range ⇑(Fin.castLEEmb ⋯)) j)
  -/
  simpa
  /-
    🎉 no goals
  -/


theorem cycleRange_of_le {n : ℕ} {i j : Fin n.succ} (h : j ≤ i) :
    cycleRange i j = if j = i then 0 else j + 1 := by
  /-
    n : Nat
    i j : Fin n.succ
    h : LE.le j i
    ⊢ Eq (i.cycleRange j) (ite (Eq j i) 0 (HAdd.hAdd j 1))
  -/
  cases n
    /-
      case zero
      i j : Fin (Nat.succ 0)
      h : LE.le j i
      ⊢ Eq (i.cycleRange j) (ite (Eq j i) 0 (HAdd.hAdd j 1))
    -/
  · subsingleton
    /-
      🎉 no goals
    -/
  have : j = (Fin.castLE (Nat.succ_le_of_lt i.is_lt))
    ⟨j, lt_of_le_of_lt h (Nat.lt_succ_self i)⟩ := by simp
  /-
    case succ
    n✝ : Nat
    i j : Fin (HAdd.hAdd n✝ 1).succ
    h : LE.le j i
    this : Eq j (Fin.castLE ⋯ ⟨↑j, ⋯⟩)
    ⊢ Eq (i.cycleRange j) (ite (Eq j i) 0 (HAdd.hAdd j 1))
  -/
  ext
  erw [this, cycleRange, ofLeftInverse'_eq_ofInjective, ←
    Function.Embedding.toEquivRange_eq_ofInjective, ← viaFintypeEmbedding,
    viaFintypeEmbedding_apply_image, Function.Embedding.coeFn_mk,
    coe_castLE, coe_finRotate]
  /-
    case succ.h
    n✝ : Nat
    i j : Fin (HAdd.hAdd n✝ 1).succ
    h : LE.le j i
    this : Eq j (Fin.castLE ⋯ ⟨↑j, ⋯⟩)
    ⊢ Eq (ite (Eq ⟨↑j, ⋯⟩ (Fin.last ↑i)) 0 (HAdd.hAdd (↑⟨↑j, ⋯⟩) 1)) ↑(ite (Eq (Fi …
  -/
  simp only [Fin.ext_iff, val_last, val_mk, val_zero, Fin.eta, castLE_mk]
  /-
    case succ.h
    n✝ : Nat
    i j : Fin (HAdd.hAdd n✝ 1).succ
    h : LE.le j i
    this : Eq j (Fin.castLE ⋯ ⟨↑j, ⋯⟩)
    ⊢ Eq (ite (Eq ↑j ↑i) 0 (HAdd.hAdd (↑j) 1)) ↑(ite (Eq ↑j ↑i) 0 (HAdd.hAdd j 1))
  -/
  split_ifs with heq
    /-
      case pos
      n✝ : Nat
      i j : Fin (HAdd.hAdd n✝ 1).succ
      h : LE.le j i
      this : Eq j (Fin.castLE ⋯ ⟨↑j, ⋯⟩)
      heq : Eq ↑j ↑i
      ⊢ Eq 0 ↑0
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      n✝ : Nat
      i j : Fin (HAdd.hAdd n✝ 1).succ
      h : LE.le j i
      this : Eq j (Fin.castLE ⋯ ⟨↑j, ⋯⟩)
      heq : Not (Eq ↑j ↑i)
      ⊢ Eq (HAdd.hAdd (↑j) 1) ↑(HAdd.hAdd j 1)
    -/
  · rw [Fin.val_add_one_of_lt]
    /-
      case neg
      n✝ : Nat
      i j : Fin (HAdd.hAdd n✝ 1).succ
      h : LE.le j i
      this : Eq j (Fin.castLE ⋯ ⟨↑j, ⋯⟩)
      heq : Not (Eq ↑j ↑i)
      ⊢ LT.lt j (Fin.last (HAdd.hAdd n✝ 1))
    -/
    exact lt_of_lt_of_le (lt_of_le_of_ne h (mt (congr_arg _) heq)) (le_last i)
    /-
      🎉 no goals
    -/


theorem coe_cycleRange_of_le {n : ℕ} {i j : Fin n.succ} (h : j ≤ i) :
    (cycleRange i j : ℕ) = if j = i then 0 else (j : ℕ) + 1 := by
  /-
    n : Nat
    i j : Fin n.succ
    h : LE.le j i
    ⊢ Eq (↑(i.cycleRange j)) (ite (Eq j i) 0 (HAdd.hAdd (↑j) 1))
  -/
  rw [cycleRange_of_le h]
  /-
    n : Nat
    i j : Fin n.succ
    h : LE.le j i
    ⊢ Eq (↑(ite (Eq j i) 0 (HAdd.hAdd j 1))) (ite (Eq j i) 0 (HAdd.hAdd (↑j) 1))
  -/
  split_ifs with h'
    /-
      case pos
      n : Nat
      i j : Fin n.succ
      h : LE.le j i
      h' : Eq j i
      ⊢ Eq (↑0) 0
    -/
  · rfl
    /-
      🎉 no goals
    -/
  exact
    val_add_one_of_lt
      (calc
        (j : ℕ) < i := Fin.lt_iff_val_lt_val.mp (lt_of_le_of_ne h h')
        _ ≤ n := Nat.lt_succ_iff.mp i.2)


theorem cycleRange_of_lt {n : ℕ} {i j : Fin n.succ} (h : j < i) : cycleRange i j = j + 1 := by
  /-
    n : Nat
    i j : Fin n.succ
    h : LT.lt j i
    ⊢ Eq (i.cycleRange j) (HAdd.hAdd j 1)
  -/
  rw [cycleRange_of_le h.le, if_neg h.ne]
  /-
    🎉 no goals
  -/


theorem coe_cycleRange_of_lt {n : ℕ} {i j : Fin n.succ} (h : j < i) :
                                       /-
                                         n : Nat
                                         i j : Fin n.succ
                                         h : LT.lt j i
                                         ⊢ Eq (↑(i.cycleRange j)) (HAdd.hAdd (↑j) 1)
                                       -/
    (cycleRange i j : ℕ) = j + 1 := by rw [coe_cycleRange_of_le h.le, if_neg h.ne]
                                       /-
                                         🎉 no goals
                                       -/


theorem cycleRange_of_eq {n : ℕ} {i j : Fin n.succ} (h : j = i) : cycleRange i j = 0 := by
  /-
    n : Nat
    i j : Fin n.succ
    h : Eq j i
    ⊢ Eq (i.cycleRange j) 0
  -/
  rw [cycleRange_of_le h.le, if_pos h]
  /-
    🎉 no goals
  -/


@[simp]
theorem cycleRange_self {n : ℕ} (i : Fin n.succ) : cycleRange i i = 0 :=
  cycleRange_of_eq rfl


theorem cycleRange_apply {n : ℕ} (i j : Fin n.succ) :
    cycleRange i j = if j < i then j + 1 else if j = i then 0 else j := by
  /-
    n : Nat
    i j : Fin n.succ
    ⊢ Eq (i.cycleRange j) (ite (LT.lt j i) (HAdd.hAdd j 1) (ite (Eq j i) 0 j))
  -/
  split_ifs with h₁ h₂
    /-
      case pos
      n : Nat
      i j : Fin n.succ
      h₁ : LT.lt j i
      ⊢ Eq (i.cycleRange j) (HAdd.hAdd j 1)
    -/
  · exact cycleRange_of_lt h₁
    /-
      🎉 no goals
    -/
    /-
      case pos
      n : Nat
      i j : Fin n.succ
      h₁ : Not (LT.lt j i)
      h₂ : Eq j i
      ⊢ Eq (i.cycleRange j) 0
    -/
  · exact cycleRange_of_eq h₂
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      i j : Fin n.succ
      h₁ : Not (LT.lt j i)
      h₂ : Not (Eq j i)
      ⊢ Eq (i.cycleRange j) j
    -/
  · exact cycleRange_of_gt (lt_of_le_of_ne (le_of_not_gt h₁) (Ne.symm h₂))
    /-
      🎉 no goals
    -/


@[simp]
theorem cycleRange_zero (n : ℕ) : cycleRange (0 : Fin n.succ) = 1 := by
  /-
    n : Nat
    ⊢ Eq (Fin.cycleRange 0) 1
  -/
  ext j
  /-
    case H.h
    n : Nat
    j : Fin n.succ
    ⊢ Eq ↑((Fin.cycleRange 0) j) ↑(1 j)
  -/
  refine Fin.cases ?_ (fun j => ?_) j
    /-
      case H.h.refine_1
      n : Nat
      j : Fin n.succ
      ⊢ Eq ↑((Fin.cycleRange 0) 0) ↑(1 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case H.h.refine_2
      n : Nat
      j✝ : Fin n.succ
      j : Fin n
      ⊢ Eq ↑((Fin.cycleRange 0) j.succ) ↑(1 j.succ)
    -/
  · rw [cycleRange_of_gt (Fin.succ_pos j), one_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem cycleRange_last (n : ℕ) : cycleRange (last n) = finRotate (n + 1) := by
  /-
    n : Nat
    ⊢ Eq (Fin.last n).cycleRange (finRotate (HAdd.hAdd n 1))
  -/
  ext i
  /-
    case H.h
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq ↑((Fin.last n).cycleRange i) ↑((finRotate (HAdd.hAdd n 1)) i)
  -/
  rw [coe_cycleRange_of_le (le_last _), coe_finRotate]
  /-
    🎉 no goals
  -/


@[simp]
theorem cycleRange_zero' {n : ℕ} (h : 0 < n) : cycleRange ⟨0, h⟩ = 1 := by
  /-
    n : Nat
    h : LT.lt 0 n
    ⊢ Eq ⟨0, h⟩.cycleRange 1
  -/
  cases' n with n
    /-
      case zero
      h : LT.lt 0 0
      ⊢ Eq ⟨0, h⟩.cycleRange 1
    -/
  · cases h
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Nat
    h : LT.lt 0 (HAdd.hAdd n 1)
    ⊢ Eq ⟨0, h⟩.cycleRange 1
  -/
  exact cycleRange_zero n
  /-
    🎉 no goals
  -/


@[simp]
theorem sign_cycleRange {n : ℕ} (i : Fin n) : Perm.sign (cycleRange i) = (-1) ^ (i : ℕ) := by
  /-
    n : Nat
    i : Fin n
    ⊢ Eq (Equiv.Perm.sign i.cycleRange) (HPow.hPow (-1) ↑i)
  -/
  simp [cycleRange]
  /-
    🎉 no goals
  -/


@[simp]
theorem succAbove_cycleRange {n : ℕ} (i j : Fin n) :
    i.succ.succAbove (i.cycleRange j) = swap 0 i.succ j.succ := by
  /-
    n : Nat
    i j : Fin n
    ⊢ Eq (i.succ.succAbove (i.cycleRange j)) ((Equiv.swap 0 i.succ) j.succ)
  -/
  cases n
    /-
      case zero
      i j : Fin 0
      ⊢ Eq (i.succ.succAbove (i.cycleRange j)) ((Equiv.swap 0 i.succ) j.succ)
    -/
  · rcases j with ⟨_, ⟨⟩⟩
    /-
      🎉 no goals
    -/
  /-
    case succ
    n✝ : Nat
    i j : Fin (HAdd.hAdd n✝ 1)
    ⊢ Eq (i.succ.succAbove (i.cycleRange j)) ((Equiv.swap 0 i.succ) j.succ)
  -/
  rcases lt_trichotomy j i with (hlt | heq | hgt)
  · have : castSucc (j + 1) = j.succ := by
      ext
      rw [coe_castSucc, val_succ, Fin.val_add_one_of_lt (lt_of_lt_of_le hlt i.le_last)]
    /-
      case succ.inl
      n✝ : Nat
      i j : Fin (HAdd.hAdd n✝ 1)
      hlt : LT.lt j i
      this : Eq (HAdd.hAdd j 1).castSucc j.succ
      ⊢ Eq (i.succ.succAbove (i.cycleRange j)) ((Equiv.swap 0 i.succ) j.succ)
    -/
    rw [Fin.cycleRange_of_lt hlt, Fin.succAbove_of_castSucc_lt, this, swap_apply_of_ne_of_ne]
      /-
        case succ.inl.a
        n✝ : Nat
        i j : Fin (HAdd.hAdd n✝ 1)
        hlt : LT.lt j i
        this : Eq (HAdd.hAdd j 1).castSucc j.succ
        ⊢ Ne j.succ 0
      -/
    · apply Fin.succ_ne_zero
      /-
        🎉 no goals
      -/
      /-
        case succ.inl.a
        n✝ : Nat
        i j : Fin (HAdd.hAdd n✝ 1)
        hlt : LT.lt j i
        this : Eq (HAdd.hAdd j 1).castSucc j.succ
        ⊢ Ne j.succ i.succ
      -/
    · exact (Fin.succ_injective _).ne hlt.ne
      /-
        🎉 no goals
      -/
      /-
        case succ.inl.h
        n✝ : Nat
        i j : Fin (HAdd.hAdd n✝ 1)
        hlt : LT.lt j i
        this : Eq (HAdd.hAdd j 1).castSucc j.succ
        ⊢ LT.lt (HAdd.hAdd j 1).castSucc i.succ
      -/
    · rw [Fin.lt_iff_val_lt_val]
      /-
        case succ.inl.h
        n✝ : Nat
        i j : Fin (HAdd.hAdd n✝ 1)
        hlt : LT.lt j i
        this : Eq (HAdd.hAdd j 1).castSucc j.succ
        ⊢ LT.lt ↑(HAdd.hAdd j 1).castSucc ↑i.succ
      -/
      simpa [this] using hlt
      /-
        🎉 no goals
      -/
    /-
      case succ.inr.inl
      n✝ : Nat
      i j : Fin (HAdd.hAdd n✝ 1)
      heq : Eq j i
      ⊢ Eq (i.succ.succAbove (i.cycleRange j)) ((Equiv.swap 0 i.succ) j.succ)
    -/
  · rw [heq, Fin.cycleRange_self, Fin.succAbove_of_castSucc_lt, swap_apply_right, Fin.castSucc_zero]
      /-
        case succ.inr.inl.h
        n✝ : Nat
        i j : Fin (HAdd.hAdd n✝ 1)
        heq : Eq j i
        ⊢ LT.lt (Fin.castSucc 0) i.succ
      -/
    · rw [Fin.castSucc_zero]
      /-
        case succ.inr.inl.h
        n✝ : Nat
        i j : Fin (HAdd.hAdd n✝ 1)
        heq : Eq j i
        ⊢ LT.lt 0 i.succ
      -/
      apply Fin.succ_pos
      /-
        🎉 no goals
      -/
    /-
      case succ.inr.inr
      n✝ : Nat
      i j : Fin (HAdd.hAdd n✝ 1)
      hgt : LT.lt i j
      ⊢ Eq (i.succ.succAbove (i.cycleRange j)) ((Equiv.swap 0 i.succ) j.succ)
    -/
  · rw [Fin.cycleRange_of_gt hgt, Fin.succAbove_of_le_castSucc, swap_apply_of_ne_of_ne]
      /-
        case succ.inr.inr.a
        n✝ : Nat
        i j : Fin (HAdd.hAdd n✝ 1)
        hgt : LT.lt i j
        ⊢ Ne j.succ 0
      -/
    · apply Fin.succ_ne_zero
      /-
        🎉 no goals
      -/
      /-
        case succ.inr.inr.a
        n✝ : Nat
        i j : Fin (HAdd.hAdd n✝ 1)
        hgt : LT.lt i j
        ⊢ Ne j.succ i.succ
      -/
    · apply (Fin.succ_injective _).ne hgt.ne.symm
      /-
        🎉 no goals
      -/
      /-
        case succ.inr.inr.h
        n✝ : Nat
        i j : Fin (HAdd.hAdd n✝ 1)
        hgt : LT.lt i j
        ⊢ LE.le i.succ j.castSucc
      -/
    · simpa [Fin.le_iff_val_le_val] using hgt
      /-
        🎉 no goals
      -/


@[simp]
theorem cycleRange_succAbove {n : ℕ} (i : Fin (n + 1)) (j : Fin n) :
    i.cycleRange (i.succAbove j) = j.succ := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    j : Fin n
    ⊢ Eq (i.cycleRange (i.succAbove j)) j.succ
  -/
  cases' lt_or_ge (castSucc j) i with h h
    /-
      case inl
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      j : Fin n
      h : LT.lt j.castSucc i
      ⊢ Eq (i.cycleRange (i.succAbove j)) j.succ
    -/
  · rw [Fin.succAbove_of_castSucc_lt _ _ h, Fin.cycleRange_of_lt h, Fin.coeSucc_eq_succ]
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      j : Fin n
      h : GE.ge j.castSucc i
      ⊢ Eq (i.cycleRange (i.succAbove j)) j.succ
    -/
  · rw [Fin.succAbove_of_le_castSucc _ _ h, Fin.cycleRange_of_gt (Fin.le_castSucc_iff.mp h)]
    /-
      🎉 no goals
    -/


@[simp]
theorem cycleRange_symm_zero {n : ℕ} (i : Fin (n + 1)) : i.cycleRange.symm 0 = i :=
                             /-
                               n : Nat
                               i : Fin (HAdd.hAdd n 1)
                               ⊢ Eq (i.cycleRange ((Equiv.symm i.cycleRange) 0)) (i.cycleRange i)
                             -/
  i.cycleRange.injective (by simp)
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem cycleRange_symm_succ {n : ℕ} (i : Fin (n + 1)) (j : Fin n) :
    i.cycleRange.symm j.succ = i.succAbove j :=
                             /-
                               n : Nat
                               i : Fin (HAdd.hAdd n 1)
                               j : Fin n
                               ⊢ Eq (i.cycleRange ((Equiv.symm i.cycleRange) j.succ)) (i.cycleRange (i.succAb …
                             -/
  i.cycleRange.injective (by simp)
                             /-
                               🎉 no goals
                             -/


theorem isCycle_cycleRange {n : ℕ} {i : Fin (n + 1)} (h0 : i ≠ 0) : IsCycle (cycleRange i) := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    h0 : Ne i 0
    ⊢ i.cycleRange.IsCycle
  -/
  cases' i with i hi
  /-
    case mk
    n i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    h0 : Ne ⟨i, hi⟩ 0
    ⊢ ⟨i, hi⟩.cycleRange.IsCycle
  -/
  cases i
    /-
      case mk.zero
      n : Nat
      hi : LT.lt 0 (HAdd.hAdd n 1)
      h0 : Ne ⟨0, hi⟩ 0
      ⊢ ⟨0, hi⟩.cycleRange.IsCycle
    -/
  · exact (h0 rfl).elim
    /-
      🎉 no goals
    -/
  /-
    case mk.succ
    n n✝ : Nat
    hi : LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd n 1)
    h0 : Ne ⟨HAdd.hAdd n✝ 1, hi⟩ 0
    ⊢ ⟨HAdd.hAdd n✝ 1, hi⟩.cycleRange.IsCycle
  -/
  exact isCycle_finRotate.extendDomain _
  /-
    🎉 no goals
  -/


@[simp]
theorem cycleType_cycleRange {n : ℕ} {i : Fin (n + 1)} (h0 : i ≠ 0) :
    cycleType (cycleRange i) = {(i + 1 : ℕ)} := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    h0 : Ne i 0
    ⊢ Eq i.cycleRange.cycleType (Singleton.singleton (HAdd.hAdd (↑i) 1))
  -/
  cases' i with i hi
  /-
    case mk
    n i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    h0 : Ne ⟨i, hi⟩ 0
    ⊢ Eq ⟨i, hi⟩.cycleRange.cycleType (Singleton.singleton (HAdd.hAdd (↑⟨i, hi⟩) 1))
  -/
  cases i
    /-
      case mk.zero
      n : Nat
      hi : LT.lt 0 (HAdd.hAdd n 1)
      h0 : Ne ⟨0, hi⟩ 0
      ⊢ Eq ⟨0, hi⟩.cycleRange.cycleType (Singleton.singleton (HAdd.hAdd (↑⟨0, hi⟩) 1))
    -/
  · exact (h0 rfl).elim
    /-
      🎉 no goals
    -/
  /-
    case mk.succ
    n n✝ : Nat
    hi : LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd n 1)
    h0 : Ne ⟨HAdd.hAdd n✝ 1, hi⟩ 0
    ⊢ Eq ⟨HAdd.hAdd n✝ 1, hi⟩.cycleRange.cycleType (Singleton.singleton (HAdd.hAdd …
  -/
  rw [cycleRange, cycleType_extendDomain]
  /-
    case mk.succ
    n n✝ : Nat
    hi : LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd n 1)
    h0 : Ne ⟨HAdd.hAdd n✝ 1, hi⟩ 0
    ⊢ Eq (finRotate (HAdd.hAdd (↑⟨HAdd.hAdd n✝ 1, hi⟩) 1)).cycleType (Singleton.si …
  -/
  exact cycleType_finRotate
  /-
    🎉 no goals
  -/


theorem isThreeCycle_cycleRange_two {n : ℕ} : IsThreeCycle (cycleRange 2 : Perm (Fin (n + 3))) := by
  /-
    n : Nat
    ⊢ (Fin.cycleRange 2).IsThreeCycle
  -/
                                              /-
                                                🎉 no goals
                                              -/
  rw [IsThreeCycle, cycleType_cycleRange] <;> simp [Fin.ext_iff]
                                              /-
                                                🎉 no goals
                                              -/


