/-- The type of Cauchy sequences of rationals with respect to the `p`-adic norm. -/
abbrev PadicSeq (p : ℕ) :=
  CauSeq _ (padicNorm p)


/-- The `p`-adic norm of the entries of a nonzero Cauchy sequence of rationals is eventually
constant. -/
theorem stationary {f : CauSeq ℚ (padicNorm p)} (hf : ¬f ≈ 0) :
    ∃ N, ∀ m n, N ≤ m → N ≤ n → padicNorm p (f n) = padicNorm p (f m) :=
  have : ∃ ε > 0, ∃ N1, ∀ j ≥ N1, ε ≤ padicNorm p (f j) :=
    CauSeq.abv_pos_of_not_limZero <| not_limZero_of_not_congr_zero hf
  let ⟨ε, hε, N1, hN1⟩ := this
  let ⟨N2, hN2⟩ := CauSeq.cauchy₂ f hε
  ⟨max N1 N2, fun n m hn hm ↦ by
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq Rat (padicNorm p)
      hf : Not (HasEquiv.Equiv f 0)
      this : Exists fun ε => And (GT.gt ε 0) (Exists fun N1 => ∀ (j : Nat), GE.ge j  …
      ε : Rat
      hε : GT.gt ε 0
      N1 : Nat
      hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le ε (padicNorm p (↑f j))
      N2 : Nat
      hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNorm p ( …
      n m : Nat
      hn : LE.le (Max.max N1 N2) n
      hm : LE.le (Max.max N1 N2) m
      ⊢ Eq (padicNorm p (↑f m)) (padicNorm p (↑f n))
    -/
    have : padicNorm p (f n - f m) < ε := hN2 _ (max_le_iff.1 hn).2 _ (max_le_iff.1 hm).2
    have : padicNorm p (f n - f m) < padicNorm p (f n) :=
      lt_of_lt_of_le this <| hN1 _ (max_le_iff.1 hn).1
    have : padicNorm p (f n - f m) < max (padicNorm p (f n)) (padicNorm p (f m)) :=
      lt_max_iff.2 (Or.inl this)
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq Rat (padicNorm p)
      hf : Not (HasEquiv.Equiv f 0)
      this✝² : Exists fun ε => And (GT.gt ε 0) (Exists fun N1 => ∀ (j : Nat), GE.ge  …
      ε : Rat
      hε : GT.gt ε 0
      N1 : Nat
      hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le ε (padicNorm p (↑f j))
      N2 : Nat
      hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNorm p ( …
      n m : Nat
      hn : LE.le (Max.max N1 N2) n
      hm : LE.le (Max.max N1 N2) m
      this✝¹ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) ε
      this✝ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) (padicNorm p (↑f n))
      this : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) (Max.max (padicNorm p (↑f …
      ⊢ Eq (padicNorm p (↑f m)) (padicNorm p (↑f n))
    -/
    by_contra hne
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq Rat (padicNorm p)
      hf : Not (HasEquiv.Equiv f 0)
      this✝² : Exists fun ε => And (GT.gt ε 0) (Exists fun N1 => ∀ (j : Nat), GE.ge  …
      ε : Rat
      hε : GT.gt ε 0
      N1 : Nat
      hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le ε (padicNorm p (↑f j))
      N2 : Nat
      hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNorm p ( …
      n m : Nat
      hn : LE.le (Max.max N1 N2) n
      hm : LE.le (Max.max N1 N2) m
      this✝¹ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) ε
      this✝ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) (padicNorm p (↑f n))
      this : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) (Max.max (padicNorm p (↑f …
      hne : Not (Eq (padicNorm p (↑f m)) (padicNorm p (↑f n)))
      ⊢ False
    -/
    rw [← padicNorm.neg (f m)] at hne
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq Rat (padicNorm p)
      hf : Not (HasEquiv.Equiv f 0)
      this✝² : Exists fun ε => And (GT.gt ε 0) (Exists fun N1 => ∀ (j : Nat), GE.ge  …
      ε : Rat
      hε : GT.gt ε 0
      N1 : Nat
      hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le ε (padicNorm p (↑f j))
      N2 : Nat
      hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNorm p ( …
      n m : Nat
      hn : LE.le (Max.max N1 N2) n
      hm : LE.le (Max.max N1 N2) m
      this✝¹ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) ε
      this✝ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) (padicNorm p (↑f n))
      this : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) (Max.max (padicNorm p (↑f …
      hne : Not (Eq (padicNorm p (Neg.neg (↑f m))) (padicNorm p (↑f n)))
      ⊢ False
    -/
    have hnam := add_eq_max_of_ne hne
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq Rat (padicNorm p)
      hf : Not (HasEquiv.Equiv f 0)
      this✝² : Exists fun ε => And (GT.gt ε 0) (Exists fun N1 => ∀ (j : Nat), GE.ge  …
      ε : Rat
      hε : GT.gt ε 0
      N1 : Nat
      hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le ε (padicNorm p (↑f j))
      N2 : Nat
      hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNorm p ( …
      n m : Nat
      hn : LE.le (Max.max N1 N2) n
      hm : LE.le (Max.max N1 N2) m
      this✝¹ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) ε
      this✝ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) (padicNorm p (↑f n))
      this : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) (Max.max (padicNorm p (↑f …
      hne : Not (Eq (padicNorm p (Neg.neg (↑f m))) (padicNorm p (↑f n)))
      hnam : Eq (padicNorm p (HAdd.hAdd (Neg.neg (↑f m)) (↑f n))) (Max.max (padicNor …
      ⊢ False
    -/
    rw [padicNorm.neg, max_comm] at hnam
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq Rat (padicNorm p)
      hf : Not (HasEquiv.Equiv f 0)
      this✝² : Exists fun ε => And (GT.gt ε 0) (Exists fun N1 => ∀ (j : Nat), GE.ge  …
      ε : Rat
      hε : GT.gt ε 0
      N1 : Nat
      hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le ε (padicNorm p (↑f j))
      N2 : Nat
      hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNorm p ( …
      n m : Nat
      hn : LE.le (Max.max N1 N2) n
      hm : LE.le (Max.max N1 N2) m
      this✝¹ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) ε
      this✝ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) (padicNorm p (↑f n))
      this : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) (Max.max (padicNorm p (↑f …
      hne : Not (Eq (padicNorm p (Neg.neg (↑f m))) (padicNorm p (↑f n)))
      hnam : Eq (padicNorm p (HAdd.hAdd (Neg.neg (↑f m)) (↑f n))) (Max.max (padicNor …
      ⊢ False
    -/
    rw [← hnam, sub_eq_add_neg, add_comm] at this
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq Rat (padicNorm p)
      hf : Not (HasEquiv.Equiv f 0)
      this✝² : Exists fun ε => And (GT.gt ε 0) (Exists fun N1 => ∀ (j : Nat), GE.ge  …
      ε : Rat
      hε : GT.gt ε 0
      N1 : Nat
      hN1 : ∀ (j : Nat), GE.ge j N1 → LE.le ε (padicNorm p (↑f j))
      N2 : Nat
      hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNorm p ( …
      n m : Nat
      hn : LE.le (Max.max N1 N2) n
      hm : LE.le (Max.max N1 N2) m
      this✝¹ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) ε
      this✝ : LT.lt (padicNorm p (HSub.hSub (↑f n) (↑f m))) (padicNorm p (↑f n))
      this : LT.lt (padicNorm p (HAdd.hAdd (Neg.neg (↑f m)) (↑f n))) (padicNorm p (H …
      hne : Not (Eq (padicNorm p (Neg.neg (↑f m))) (padicNorm p (↑f n)))
      hnam : Eq (padicNorm p (HAdd.hAdd (Neg.neg (↑f m)) (↑f n))) (Max.max (padicNor …
      ⊢ False
    -/
    apply _root_.lt_irrefl _ this⟩
    /-
      🎉 no goals
    -/


/-- For all `n ≥ stationaryPoint f hf`, the `p`-adic norm of `f n` is the same. -/
def stationaryPoint {f : PadicSeq p} (hf : ¬f ≈ 0) : ℕ :=
  Classical.choose <| stationary hf


theorem stationaryPoint_spec {f : PadicSeq p} (hf : ¬f ≈ 0) :
    ∀ {m n},
      stationaryPoint hf ≤ m → stationaryPoint hf ≤ n → padicNorm p (f n) = padicNorm p (f m) :=
  @(Classical.choose_spec <| stationary hf)


open Classical in
/-- Since the norm of the entries of a Cauchy sequence is eventually stationary,
we can lift the norm to sequences. -/
def norm (f : PadicSeq p) : ℚ :=
  if hf : f ≈ 0 then 0 else padicNorm p (f (stationaryPoint hf))


theorem norm_zero_iff (f : PadicSeq p) : f.norm = 0 ↔ f ≈ 0 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    ⊢ Iff (Eq f.norm 0) (HasEquiv.Equiv f 0)
  -/
  constructor
    /-
      case mp
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      ⊢ Eq f.norm 0 → HasEquiv.Equiv f 0
    -/
  · intro h
    /-
      case mp
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      h : Eq f.norm 0
      ⊢ HasEquiv.Equiv f 0
    -/
    by_contra hf
    /-
      case mp
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      h : Eq f.norm 0
      hf : Not (HasEquiv.Equiv f 0)
      ⊢ False
    -/
    unfold norm at h
    /-
      case mp
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      h : Eq (dite (HasEquiv.Equiv f 0) (fun hf => 0) fun hf => padicNorm p (↑f (Pad …
      hf : Not (HasEquiv.Equiv f 0)
      ⊢ False
    -/
    split_ifs at h
    /-
      case mp
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      h : Eq (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) 0
      ⊢ False
    -/
    apply hf
    /-
      case mp
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      h : Eq (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) 0
      ⊢ HasEquiv.Equiv f 0
    -/
    intro ε hε
    /-
      case mp
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      h : Eq (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) 0
      ε : Rat
      hε : GT.gt ε 0
      ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (↑(HSub.hSub f 0 …
    -/
    exists stationaryPoint hf
    /-
      case mp
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      h : Eq (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) 0
      ε : Rat
      hε : GT.gt ε 0
      ⊢ ∀ (j : Nat), GE.ge j (PadicSeq.stationaryPoint hf) → LT.lt (padicNorm p (↑(H …
    -/
    intro j hj
    /-
      case mp
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      h : Eq (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) 0
      ε : Rat
      hε : GT.gt ε 0
      j : Nat
      hj : GE.ge j (PadicSeq.stationaryPoint hf)
      ⊢ LT.lt (padicNorm p (↑(HSub.hSub f 0) j)) ε
    -/
    have heq := stationaryPoint_spec hf le_rfl hj
    /-
      case mp
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      h : Eq (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) 0
      ε : Rat
      hε : GT.gt ε 0
      j : Nat
      hj : GE.ge j (PadicSeq.stationaryPoint hf)
      heq : Eq (padicNorm p (↑f j)) (padicNorm p (↑f (PadicSeq.stationaryPoint hf)))
      ⊢ LT.lt (padicNorm p (↑(HSub.hSub f 0) j)) ε
    -/
    simpa [h, heq]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      ⊢ HasEquiv.Equiv f 0 → Eq f.norm 0
    -/
  · intro h
    /-
      case mpr
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      h : HasEquiv.Equiv f 0
      ⊢ Eq f.norm 0
    -/
    simp [norm, h]
    /-
      🎉 no goals
    -/


theorem equiv_zero_of_val_eq_of_equiv_zero {f g : PadicSeq p}
    (h : ∀ k, padicNorm p (f k) = padicNorm p (g k)) (hf : f ≈ 0) : g ≈ 0 := fun ε hε ↦
  let ⟨i, hi⟩ := hf _ hε
                    /-
                      p : Nat
                      inst✝ : Fact (Nat.Prime p)
                      f g : PadicSeq p
                      h : ∀ (k : Nat), Eq (padicNorm p (↑f k)) (padicNorm p (↑g k))
                      hf : HasEquiv.Equiv f 0
                      ε : Rat
                      hε : GT.gt ε 0
                      i : Nat
                      hi : ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (↑(HSub.hSub f 0) j)) ε
                      j : Nat
                      hj : GE.ge j i
                      ⊢ LT.lt (padicNorm p (↑(HSub.hSub g 0) j)) ε
                    -/
  ⟨i, fun j hj ↦ by simpa [h] using hi _ hj⟩
                    /-
                      🎉 no goals
                    -/


theorem norm_nonzero_of_not_equiv_zero {f : PadicSeq p} (hf : ¬f ≈ 0) : f.norm ≠ 0 :=
  hf ∘ f.norm_zero_iff.1


theorem norm_eq_norm_app_of_nonzero {f : PadicSeq p} (hf : ¬f ≈ 0) :
    ∃ k, f.norm = padicNorm p k ∧ k ≠ 0 :=
                                                                  /-
                                                                    p : Nat
                                                                    inst✝ : Fact (Nat.Prime p)
                                                                    f : PadicSeq p
                                                                    hf : Not (HasEquiv.Equiv f 0)
                                                                    ⊢ Eq f.norm (padicNorm p (↑f (PadicSeq.stationaryPoint hf)))
                                                                  -/
  have heq : f.norm = padicNorm p (f <| stationaryPoint hf) := by simp [norm, hf]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  ⟨f <| stationaryPoint hf, heq, fun h ↦
                                          /-
                                            p : Nat
                                            inst✝ : Fact (Nat.Prime p)
                                            f : PadicSeq p
                                            hf : Not (HasEquiv.Equiv f 0)
                                            heq : Eq f.norm (padicNorm p (↑f (PadicSeq.stationaryPoint hf)))
                                            h : Eq (↑f (PadicSeq.stationaryPoint hf)) 0
                                            ⊢ Eq f.norm 0
                                          -/
    norm_nonzero_of_not_equiv_zero hf (by simpa [h] using heq)⟩
                                          /-
                                            🎉 no goals
                                          -/


theorem not_limZero_const_of_nonzero {q : ℚ} (hq : q ≠ 0) : ¬LimZero (const (padicNorm p) q) :=
  fun h' ↦ hq <| const_limZero.1 h'


theorem not_equiv_zero_const_of_nonzero {q : ℚ} (hq : q ≠ 0) : ¬const (padicNorm p) q ≈ 0 :=
  fun h : LimZero (const (padicNorm p) q - 0) ↦
                                                   /-
                                                     p : Nat
                                                     inst✝ : Fact (Nat.Prime p)
                                                     q : Rat
                                                     hq : Ne q 0
                                                     h : (HSub.hSub (CauSeq.const (padicNorm p) q) 0).LimZero
                                                     ⊢ (CauSeq.const (padicNorm p) q).LimZero
                                                   -/
    not_limZero_const_of_nonzero (p := p) hq <| by simpa using h
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem norm_nonneg (f : PadicSeq p) : 0 ≤ f.norm := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    ⊢ LE.le 0 f.norm
  -/
  classical exact if hf : f ≈ 0 then by simp [hf, norm] else by simp [norm, hf, padicNorm.nonneg]
  /-
    🎉 no goals
  -/


/-- An auxiliary lemma for manipulating sequence indices. -/
theorem lift_index_left_left {f : PadicSeq p} (hf : ¬f ≈ 0) (v2 v3 : ℕ) :
    padicNorm p (f (stationaryPoint hf)) =
    padicNorm p (f (max (stationaryPoint hf) (max v2 v3))) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    v2 v3 : Nat
    ⊢ Eq (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (padicNorm p (↑f (Max.ma …
  -/
  apply stationaryPoint_spec hf
    /-
      case a
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      v2 v3 : Nat
      ⊢ LE.le (PadicSeq.stationaryPoint hf) (Max.max (PadicSeq.stationaryPoint hf) ( …
    -/
  · apply le_max_left
    /-
      🎉 no goals
    -/
    /-
      case a
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      v2 v3 : Nat
      ⊢ LE.le (PadicSeq.stationaryPoint hf) (PadicSeq.stationaryPoint hf)
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/


/-- An auxiliary lemma for manipulating sequence indices. -/
theorem lift_index_left {f : PadicSeq p} (hf : ¬f ≈ 0) (v1 v3 : ℕ) :
    padicNorm p (f (stationaryPoint hf)) =
    padicNorm p (f (max v1 (max (stationaryPoint hf) v3))) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    v1 v3 : Nat
    ⊢ Eq (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (padicNorm p (↑f (Max.ma …
  -/
  apply stationaryPoint_spec hf
    /-
      case a
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      v1 v3 : Nat
      ⊢ LE.le (PadicSeq.stationaryPoint hf) (Max.max v1 (Max.max (PadicSeq.stationar …
    -/
  · apply le_trans
      /-
        case a.a
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        f : PadicSeq p
        hf : Not (HasEquiv.Equiv f 0)
        v1 v3 : Nat
        ⊢ LE.le (PadicSeq.stationaryPoint hf) ?a.b✝
      -/
    · apply le_max_left _ v3
      /-
        🎉 no goals
      -/
      /-
        case a.a
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        f : PadicSeq p
        hf : Not (HasEquiv.Equiv f 0)
        v1 v3 : Nat
        ⊢ LE.le (Max.max (PadicSeq.stationaryPoint hf) v3) (Max.max v1 (Max.max (Padic …
      -/
    · apply le_max_right
      /-
        🎉 no goals
      -/
    /-
      case a
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      v1 v3 : Nat
      ⊢ LE.le (PadicSeq.stationaryPoint hf) (PadicSeq.stationaryPoint hf)
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/


/-- An auxiliary lemma for manipulating sequence indices. -/
theorem lift_index_right {f : PadicSeq p} (hf : ¬f ≈ 0) (v1 v2 : ℕ) :
    padicNorm p (f (stationaryPoint hf)) =
    padicNorm p (f (max v1 (max v2 (stationaryPoint hf)))) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    v1 v2 : Nat
    ⊢ Eq (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (padicNorm p (↑f (Max.ma …
  -/
  apply stationaryPoint_spec hf
    /-
      case a
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      v1 v2 : Nat
      ⊢ LE.le (PadicSeq.stationaryPoint hf) (Max.max v1 (Max.max v2 (PadicSeq.statio …
    -/
  · apply le_trans
      /-
        case a.a
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        f : PadicSeq p
        hf : Not (HasEquiv.Equiv f 0)
        v1 v2 : Nat
        ⊢ LE.le (PadicSeq.stationaryPoint hf) ?a.b✝
      -/
    · apply le_max_right v2
      /-
        🎉 no goals
      -/
      /-
        case a.a
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        f : PadicSeq p
        hf : Not (HasEquiv.Equiv f 0)
        v1 v2 : Nat
        ⊢ LE.le (Max.max v2 (PadicSeq.stationaryPoint hf)) (Max.max v1 (Max.max v2 (Pa …
      -/
    · apply le_max_right
      /-
        🎉 no goals
      -/
    /-
      case a
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      v1 v2 : Nat
      ⊢ LE.le (PadicSeq.stationaryPoint hf) (PadicSeq.stationaryPoint hf)
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/


open Classical in
/-- The `p`-adic valuation on `ℚ` lifts to `PadicSeq p`.
`Valuation f` is defined to be the valuation of the (`ℚ`-valued) stationary point of `f`. -/
def valuation (f : PadicSeq p) : ℤ :=
  if hf : f ≈ 0 then 0 else padicValRat p (f (stationaryPoint hf))


theorem norm_eq_zpow_neg_valuation {f : PadicSeq p} (hf : ¬f ≈ 0) :
    f.norm = (p : ℚ) ^ (-f.valuation : ℤ) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    ⊢ Eq f.norm (HPow.hPow (↑p) (Neg.neg f.valuation))
  -/
  rw [norm, valuation, dif_neg hf, dif_neg hf, padicNorm, if_neg]
  /-
    case hnc
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    ⊢ Not (Eq (↑f (PadicSeq.stationaryPoint hf)) 0)
  -/
  intro H
  /-
    case hnc
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    H : Eq (↑f (PadicSeq.stationaryPoint hf)) 0
    ⊢ False
  -/
  apply CauSeq.not_limZero_of_not_congr_zero hf
  /-
    case hnc
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    H : Eq (↑f (PadicSeq.stationaryPoint hf)) 0
    ⊢ CauSeq.LimZero f
  -/
  intro ε hε
  /-
    case hnc
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    H : Eq (↑f (PadicSeq.stationaryPoint hf)) 0
    ε : Rat
    hε : GT.gt ε 0
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (↑f j)) ε
  -/
  use stationaryPoint hf
  /-
    case h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    H : Eq (↑f (PadicSeq.stationaryPoint hf)) 0
    ε : Rat
    hε : GT.gt ε 0
    ⊢ ∀ (j : Nat), GE.ge j (PadicSeq.stationaryPoint hf) → LT.lt (padicNorm p (↑f  …
  -/
  intro n hn
  /-
    case h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    H : Eq (↑f (PadicSeq.stationaryPoint hf)) 0
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : GE.ge n (PadicSeq.stationaryPoint hf)
    ⊢ LT.lt (padicNorm p (↑f n)) ε
  -/
  rw [stationaryPoint_spec hf le_rfl hn]
  /-
    case h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    H : Eq (↑f (PadicSeq.stationaryPoint hf)) 0
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : GE.ge n (PadicSeq.stationaryPoint hf)
    ⊢ LT.lt (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) ε
  -/
  simpa [H] using hε
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-10")] alias norm_eq_pow_val := norm_eq_zpow_neg_valuation


theorem val_eq_iff_norm_eq {f g : PadicSeq p} (hf : ¬f ≈ 0) (hg : ¬g ≈ 0) :
    f.valuation = g.valuation ↔ f.norm = g.norm := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    ⊢ Iff (Eq f.valuation g.valuation) (Eq f.norm g.norm)
  -/
  rw [norm_eq_zpow_neg_valuation hf, norm_eq_zpow_neg_valuation hg, ← neg_inj, zpow_right_inj₀]
    /-
      case ha₀
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f g : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      hg : Not (HasEquiv.Equiv g 0)
      ⊢ LT.lt 0 ↑p
    -/
  · exact mod_cast (Fact.out : p.Prime).pos
    /-
      🎉 no goals
    -/
    /-
      case ha₁
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f g : PadicSeq p
      hf : Not (HasEquiv.Equiv f 0)
      hg : Not (HasEquiv.Equiv g 0)
      ⊢ Ne (↑p) 1
    -/
  · exact mod_cast (Fact.out : p.Prime).ne_one
    /-
      🎉 no goals
    -/


theorem norm_mul (f g : PadicSeq p) : (f * g).norm = f.norm * g.norm := by
  classical
  exact if hf : f ≈ 0 then by
    have hg : f * g ≈ 0 := mul_equiv_zero' _ hf
    simp only [hf, hg, norm, dif_pos, zero_mul]
  else
    if hg : g ≈ 0 then by
      have hf : f * g ≈ 0 := mul_equiv_zero _ hg
      simp only [hf, hg, norm, dif_pos, mul_zero]
    else by
      unfold norm
      have hfg := mul_not_equiv_zero hf hg
      simp only [hfg, hf, hg, dite_false]
      -- Porting note: originally `padic_index_simp [hfg, hf, hg]`
      rw [lift_index_left_left hfg, lift_index_left hf, lift_index_right hg]
      apply padicNorm.mul


theorem eq_zero_iff_equiv_zero (f : PadicSeq p) : mk f = 0 ↔ f ≈ 0 :=
  mk_eq


theorem ne_zero_iff_nequiv_zero (f : PadicSeq p) : mk f ≠ 0 ↔ ¬f ≈ 0 :=
  eq_zero_iff_equiv_zero _ |>.not


theorem norm_const (q : ℚ) : norm (const (padicNorm p) q) = padicNorm p q :=
  if hq : q = 0 then by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Rat
      hq : Eq q 0
      ⊢ Eq (PadicSeq.norm (CauSeq.const (padicNorm p) q)) (padicNorm p q)
    -/
    have : const (padicNorm p) q ≈ 0 := by simpa [hq] using Setoid.refl (const (padicNorm p) 0)
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Rat
      hq : Eq q 0
      this : HasEquiv.Equiv (CauSeq.const (padicNorm p) q) 0
      ⊢ Eq (PadicSeq.norm (CauSeq.const (padicNorm p) q)) (padicNorm p q)
    -/
    subst hq; simp [norm, this]
              /-
                🎉 no goals
              -/
  else by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Rat
      hq : Not (Eq q 0)
      ⊢ Eq (PadicSeq.norm (CauSeq.const (padicNorm p) q)) (padicNorm p q)
    -/
    have : ¬const (padicNorm p) q ≈ 0 := not_equiv_zero_const_of_nonzero hq
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Rat
      hq : Not (Eq q 0)
      this : Not (HasEquiv.Equiv (CauSeq.const (padicNorm p) q) 0)
      ⊢ Eq (PadicSeq.norm (CauSeq.const (padicNorm p) q)) (padicNorm p q)
    -/
    simp [norm, this]
    /-
      🎉 no goals
    -/


theorem norm_values_discrete (a : PadicSeq p) (ha : ¬a ≈ 0) : ∃ z : ℤ, a.norm = (p : ℚ) ^ (-z) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    a : PadicSeq p
    ha : Not (HasEquiv.Equiv a 0)
    ⊢ Exists fun z => Eq a.norm (HPow.hPow (↑p) (Neg.neg z))
  -/
  let ⟨k, hk, hk'⟩ := norm_eq_norm_app_of_nonzero ha
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    a : PadicSeq p
    ha : Not (HasEquiv.Equiv a 0)
    k : Rat
    hk : Eq a.norm (padicNorm p k)
    hk' : Ne k 0
    ⊢ Exists fun z => Eq a.norm (HPow.hPow (↑p) (Neg.neg z))
  -/
  simpa [hk] using padicNorm.values_discrete hk'
  /-
    🎉 no goals
  -/


theorem norm_one : norm (1 : PadicSeq p) = 1 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (PadicSeq.norm 1) 1
  -/
  have h1 : ¬(1 : PadicSeq p) ≈ 0 := one_not_equiv_zero _
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    h1 : Not (HasEquiv.Equiv 1 0)
    ⊢ Eq (PadicSeq.norm 1) 1
  -/
  simp [h1, norm, hp.1.one_lt]
  /-
    🎉 no goals
  -/


private theorem norm_eq_of_equiv_aux {f g : PadicSeq p} (hf : ¬f ≈ 0) (hg : ¬g ≈ 0) (hfg : f ≈ g)
    (h : padicNorm p (f (stationaryPoint hf)) ≠ padicNorm p (g (stationaryPoint hg)))
    (hlt : padicNorm p (g (stationaryPoint hg)) < padicNorm p (f (stationaryPoint hf))) :
    False := by
  have hpn : 0 < padicNorm p (f (stationaryPoint hf)) - padicNorm p (g (stationaryPoint hg)) :=
    sub_pos_of_lt hlt
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    hfg : HasEquiv.Equiv f g
    h : Ne (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (padicNorm p (↑g (Padi …
    hlt : LT.lt (padicNorm p (↑g (PadicSeq.stationaryPoint hg))) (padicNorm p (↑f  …
    hpn : LT.lt 0 (HSub.hSub (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (pad …
    ⊢ False
  -/
  cases' hfg _ hpn with N hN
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    hfg : HasEquiv.Equiv f g
    h : Ne (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (padicNorm p (↑g (Padi …
    hlt : LT.lt (padicNorm p (↑g (PadicSeq.stationaryPoint hg))) (padicNorm p (↑f  …
    hpn : LT.lt 0 (HSub.hSub (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (pad …
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (padicNorm p (↑(HSub.hSub f g) j)) (HSub.h …
    ⊢ False
  -/
  let i := max N (max (stationaryPoint hf) (stationaryPoint hg))
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    hfg : HasEquiv.Equiv f g
    h : Ne (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (padicNorm p (↑g (Padi …
    hlt : LT.lt (padicNorm p (↑g (PadicSeq.stationaryPoint hg))) (padicNorm p (↑f  …
    hpn : LT.lt 0 (HSub.hSub (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (pad …
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (padicNorm p (↑(HSub.hSub f g) j)) (HSub.h …
    i : Nat := Max.max N (Max.max (PadicSeq.stationaryPoint hf) (PadicSeq.stationa …
    ⊢ False
  -/
  have hi : N ≤ i := le_max_left _ _
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    hfg : HasEquiv.Equiv f g
    h : Ne (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (padicNorm p (↑g (Padi …
    hlt : LT.lt (padicNorm p (↑g (PadicSeq.stationaryPoint hg))) (padicNorm p (↑f  …
    hpn : LT.lt 0 (HSub.hSub (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (pad …
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (padicNorm p (↑(HSub.hSub f g) j)) (HSub.h …
    i : Nat := Max.max N (Max.max (PadicSeq.stationaryPoint hf) (PadicSeq.stationa …
    hi : LE.le N i
    ⊢ False
  -/
  have hN' := hN _ hi
  -- Porting note: originally `padic_index_simp [N, hf, hg] at hN' h hlt`
  rw [lift_index_left hf N (stationaryPoint hg), lift_index_right hg N (stationaryPoint hf)]
    at hN' h hlt
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    hfg : HasEquiv.Equiv f g
    hpn : LT.lt 0 (HSub.hSub (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (pad …
    N : Nat
    hlt : LT.lt (padicNorm p (↑g (Max.max N (Max.max (PadicSeq.stationaryPoint hf) …
    h : Ne (padicNorm p (↑f (Max.max N (Max.max (PadicSeq.stationaryPoint hf) (Pad …
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (padicNorm p (↑(HSub.hSub f g) j)) (HSub.h …
    i : Nat := Max.max N (Max.max (PadicSeq.stationaryPoint hf) (PadicSeq.stationa …
    hi : LE.le N i
    hN' : LT.lt (padicNorm p (↑(HSub.hSub f g) i)) (HSub.hSub (padicNorm p (↑f (Ma …
    ⊢ False
  -/
  have hpne : padicNorm p (f i) ≠ padicNorm p (-g i) := by rwa [← padicNorm.neg (g i)] at h
  rw [CauSeq.sub_apply, sub_eq_add_neg, add_eq_max_of_ne hpne, padicNorm.neg, max_eq_left_of_lt hlt]
    at hN'
  have : padicNorm p (f i) < padicNorm p (f i) := by
    apply lt_of_lt_of_le hN'
    apply sub_le_self
    apply padicNorm.nonneg
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    hfg : HasEquiv.Equiv f g
    hpn : LT.lt 0 (HSub.hSub (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (pad …
    N : Nat
    hlt : LT.lt (padicNorm p (↑g (Max.max N (Max.max (PadicSeq.stationaryPoint hf) …
    h : Ne (padicNorm p (↑f (Max.max N (Max.max (PadicSeq.stationaryPoint hf) (Pad …
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (padicNorm p (↑(HSub.hSub f g) j)) (HSub.h …
    i : Nat := Max.max N (Max.max (PadicSeq.stationaryPoint hf) (PadicSeq.stationa …
    hi : LE.le N i
    hN' : LT.lt (padicNorm p (↑f (Max.max N (Max.max (PadicSeq.stationaryPoint hf) …
    hpne : Ne (padicNorm p (↑f i)) (padicNorm p (Neg.neg (↑g i)))
    this : LT.lt (padicNorm p (↑f i)) (padicNorm p (↑f i))
    ⊢ False
  -/
  exact lt_irrefl _ this
  /-
    🎉 no goals
  -/


private theorem norm_eq_of_equiv {f g : PadicSeq p} (hf : ¬f ≈ 0) (hg : ¬g ≈ 0) (hfg : f ≈ g) :
    padicNorm p (f (stationaryPoint hf)) = padicNorm p (g (stationaryPoint hg)) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    hfg : HasEquiv.Equiv f g
    ⊢ Eq (padicNorm p (↑f (PadicSeq.stationaryPoint hf))) (padicNorm p (↑g (PadicS …
  -/
  by_contra h
  cases lt_or_le (padicNorm p (g (stationaryPoint hg))) (padicNorm p (f (stationaryPoint hf))) with
  | inl hlt =>
    exact norm_eq_of_equiv_aux hf hg hfg h hlt
  | inr hle =>
    apply norm_eq_of_equiv_aux hg hf (Setoid.symm hfg) (Ne.symm h)
    exact lt_of_le_of_ne hle h


theorem norm_equiv {f g : PadicSeq p} (hfg : f ≈ g) : f.norm = g.norm := by
  classical
  exact if hf : f ≈ 0 then by
    have hg : g ≈ 0 := Setoid.trans (Setoid.symm hfg) hf
    simp [norm, hf, hg]
  else by
    have hg : ¬g ≈ 0 := hf ∘ Setoid.trans hfg
    unfold norm; split_ifs; exact norm_eq_of_equiv hf hg hfg


private theorem norm_nonarchimedean_aux {f g : PadicSeq p} (hfg : ¬f + g ≈ 0) (hf : ¬f ≈ 0)
    (hg : ¬g ≈ 0) : (f + g).norm ≤ max f.norm g.norm := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    hfg : Not (HasEquiv.Equiv (HAdd.hAdd f g) 0)
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    ⊢ LE.le (HAdd.hAdd f g).norm (Max.max f.norm g.norm)
  -/
  unfold norm; split_ifs
  -- Porting note: originally `padic_index_simp [hfg, hf, hg]`
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    hfg : Not (HasEquiv.Equiv (HAdd.hAdd f g) 0)
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    ⊢ LE.le (padicNorm p (↑(HAdd.hAdd f g) (PadicSeq.stationaryPoint hfg))) (Max.m …
  -/
  rw [lift_index_left_left hfg, lift_index_left hf, lift_index_right hg]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    hfg : Not (HasEquiv.Equiv (HAdd.hAdd f g) 0)
    hf : Not (HasEquiv.Equiv f 0)
    hg : Not (HasEquiv.Equiv g 0)
    ⊢ LE.le (padicNorm p (↑(HAdd.hAdd f g) (Max.max (PadicSeq.stationaryPoint hfg) …
  -/
  apply padicNorm.nonarchimedean
  /-
    🎉 no goals
  -/


theorem norm_nonarchimedean (f g : PadicSeq p) : (f + g).norm ≤ max f.norm g.norm := by
  classical
  exact if hfg : f + g ≈ 0 then by
    have : 0 ≤ max f.norm g.norm := le_max_of_le_left (norm_nonneg _)
    simpa only [hfg, norm]
  else
    if hf : f ≈ 0 then by
      have hfg' : f + g ≈ g := by
        change LimZero (f - 0) at hf
        show LimZero (f + g - g); · simpa only [sub_zero, add_sub_cancel_right] using hf
      have hcfg : (f + g).norm = g.norm := norm_equiv hfg'
      have hcl : f.norm = 0 := (norm_zero_iff f).2 hf
      have : max f.norm g.norm = g.norm := by rw [hcl]; exact max_eq_right (norm_nonneg _)
      rw [this, hcfg]
    else
      if hg : g ≈ 0 then by
        have hfg' : f + g ≈ f := by
          change LimZero (g - 0) at hg
          show LimZero (f + g - f); · simpa only [add_sub_cancel_left, sub_zero] using hg
        have hcfg : (f + g).norm = f.norm := norm_equiv hfg'
        have hcl : g.norm = 0 := (norm_zero_iff g).2 hg
        have : max f.norm g.norm = f.norm := by rw [hcl]; exact max_eq_left (norm_nonneg _)
        rw [this, hcfg]
      else norm_nonarchimedean_aux hfg hf hg


theorem norm_eq {f g : PadicSeq p} (h : ∀ k, padicNorm p (f k) = padicNorm p (g k)) :
    f.norm = g.norm := by
  classical
  exact if hf : f ≈ 0 then by
    have hg : g ≈ 0 := equiv_zero_of_val_eq_of_equiv_zero h hf
    simp only [hf, hg, norm, dif_pos]
  else by
    have hg : ¬g ≈ 0 := fun hg ↦
      hf <| equiv_zero_of_val_eq_of_equiv_zero (by simp only [h, forall_const, eq_self_iff_true]) hg
    simp only [hg, hf, norm, dif_neg, not_false_iff]
    let i := max (stationaryPoint hf) (stationaryPoint hg)
    have hpf : padicNorm p (f (stationaryPoint hf)) = padicNorm p (f i) := by
      apply stationaryPoint_spec
      · apply le_max_left
      · exact le_rfl
    have hpg : padicNorm p (g (stationaryPoint hg)) = padicNorm p (g i) := by
      apply stationaryPoint_spec
      · apply le_max_right
      · exact le_rfl
    rw [hpf, hpg, h]


theorem norm_neg (a : PadicSeq p) : (-a).norm = a.norm :=
                /-
                  p : Nat
                  hp : Fact (Nat.Prime p)
                  a : PadicSeq p
                  ⊢ ∀ (k : Nat), Eq (padicNorm p (↑(Neg.neg a) k)) (padicNorm p (↑a k))
                -/
  norm_eq <| by simp
                /-
                  🎉 no goals
                -/


theorem norm_eq_of_add_equiv_zero {f g : PadicSeq p} (h : f + g ≈ 0) : f.norm = g.norm := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    h : HasEquiv.Equiv (HAdd.hAdd f g) 0
    ⊢ Eq f.norm g.norm
  -/
  have : LimZero (f + g - 0) := h
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    h : HasEquiv.Equiv (HAdd.hAdd f g) 0
    this : (HSub.hSub (HAdd.hAdd f g) 0).LimZero
    ⊢ Eq f.norm g.norm
  -/
  have : f ≈ -g := show LimZero (f - -g) by simpa only [sub_zero, sub_neg_eq_add]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    h : HasEquiv.Equiv (HAdd.hAdd f g) 0
    this✝ : (HSub.hSub (HAdd.hAdd f g) 0).LimZero
    this : HasEquiv.Equiv f (Neg.neg g)
    ⊢ Eq f.norm g.norm
  -/
  have : f.norm = (-g).norm := norm_equiv this
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    f g : PadicSeq p
    h : HasEquiv.Equiv (HAdd.hAdd f g) 0
    this✝¹ : (HSub.hSub (HAdd.hAdd f g) 0).LimZero
    this✝ : HasEquiv.Equiv f (Neg.neg g)
    this : Eq f.norm (Neg.neg g).norm
    ⊢ Eq f.norm g.norm
  -/
  simpa only [norm_neg] using this
  /-
    🎉 no goals
  -/


theorem add_eq_max_of_ne {f g : PadicSeq p} (hfgne : f.norm ≠ g.norm) :
    (f + g).norm = max f.norm g.norm := by
  classical
  have hfg : ¬f + g ≈ 0 := mt norm_eq_of_add_equiv_zero hfgne
  exact if hf : f ≈ 0 then by
    have : LimZero (f - 0) := hf
    have : f + g ≈ g := show LimZero (f + g - g) by simpa only [sub_zero, add_sub_cancel_right]
    have h1 : (f + g).norm = g.norm := norm_equiv this
    have h2 : f.norm = 0 := (norm_zero_iff _).2 hf
    rw [h1, h2, max_eq_right (norm_nonneg _)]
  else
    if hg : g ≈ 0 then by
      have : LimZero (g - 0) := hg
      have : f + g ≈ f := show LimZero (f + g - f) by simpa only [add_sub_cancel_left, sub_zero]
      have h1 : (f + g).norm = f.norm := norm_equiv this
      have h2 : g.norm = 0 := (norm_zero_iff _).2 hg
      rw [h1, h2, max_eq_left (norm_nonneg _)]
    else by
      unfold norm at hfgne ⊢; split_ifs at hfgne ⊢
      -- Porting note: originally `padic_index_simp [hfg, hf, hg] at hfgne ⊢`
      rw [lift_index_left hf, lift_index_right hg] at hfgne
      · rw [lift_index_left_left hfg, lift_index_left hf, lift_index_right hg]
        exact padicNorm.add_eq_max_of_ne hfgne


/-- The `p`-adic numbers `ℚ_[p]` are the Cauchy completion of `ℚ` with respect to the `p`-adic norm.
-/
def Padic (p : ℕ) [Fact p.Prime] :=
  CauSeq.Completion.Cauchy (padicNorm p)


/-- notation for p-padic rationals -/
notation "ℚ_[" p "]" => Padic p


instance field : Field ℚ_[p] :=
  Cauchy.field


instance : Inhabited ℚ_[p] :=
  ⟨0⟩

-- short circuits

instance : CommRing ℚ_[p] :=
  Cauchy.commRing


instance : Ring ℚ_[p] :=
  Cauchy.ring


                            /-
                              p : Nat
                              inst✝ : Fact (Nat.Prime p)
                              ⊢ Zero (Padic p)
                            -/
instance : Zero ℚ_[p] := by infer_instance
                            /-
                              🎉 no goals
                            -/


                           /-
                             p : Nat
                             inst✝ : Fact (Nat.Prime p)
                             ⊢ One (Padic p)
                           -/
instance : One ℚ_[p] := by infer_instance
                           /-
                             🎉 no goals
                           -/


                           /-
                             p : Nat
                             inst✝ : Fact (Nat.Prime p)
                             ⊢ Add (Padic p)
                           -/
instance : Add ℚ_[p] := by infer_instance
                           /-
                             🎉 no goals
                           -/


                           /-
                             p : Nat
                             inst✝ : Fact (Nat.Prime p)
                             ⊢ Mul (Padic p)
                           -/
instance : Mul ℚ_[p] := by infer_instance
                           /-
                             🎉 no goals
                           -/


                           /-
                             p : Nat
                             inst✝ : Fact (Nat.Prime p)
                             ⊢ Sub (Padic p)
                           -/
instance : Sub ℚ_[p] := by infer_instance
                           /-
                             🎉 no goals
                           -/


                           /-
                             p : Nat
                             inst✝ : Fact (Nat.Prime p)
                             ⊢ Neg (Padic p)
                           -/
instance : Neg ℚ_[p] := by infer_instance
                           /-
                             🎉 no goals
                           -/


                           /-
                             p : Nat
                             inst✝ : Fact (Nat.Prime p)
                             ⊢ Div (Padic p)
                           -/
instance : Div ℚ_[p] := by infer_instance
                           /-
                             🎉 no goals
                           -/


                                    /-
                                      p : Nat
                                      inst✝ : Fact (Nat.Prime p)
                                      ⊢ AddCommGroup (Padic p)
                                    -/
instance : AddCommGroup ℚ_[p] := by infer_instance
                                    /-
                                      🎉 no goals
                                    -/


/-- Builds the equivalence class of a Cauchy sequence of rationals. -/
def mk : PadicSeq p → ℚ_[p] :=
  Quotient.mk'


theorem zero_def : (0 : ℚ_[p]) = ⟦0⟧ := rfl


theorem mk_eq {f g : PadicSeq p} : mk f = mk g ↔ f ≈ g :=
  Quotient.eq'


theorem const_equiv {q r : ℚ} : const (padicNorm p) q ≈ const (padicNorm p) r ↔ q = r :=
  ⟨fun heq ↦ eq_of_sub_eq_zero <| const_limZero.1 heq, fun heq ↦ by
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      q r : Rat
      heq : Eq q r
      ⊢ HasEquiv.Equiv (CauSeq.const (padicNorm p) q) (CauSeq.const (padicNorm p) r)
    -/
    rw [heq]⟩
    /-
      🎉 no goals
    -/


@[norm_cast]
theorem coe_inj {q r : ℚ} : (↑q : ℚ_[p]) = ↑r ↔ q = r :=
                                                  /-
                                                    p : Nat
                                                    inst✝ : Fact (Nat.Prime p)
                                                    q r : Rat
                                                    h : Eq q r
                                                    ⊢ Eq ↑q ↑r
                                                  -/
  ⟨(const_equiv p).1 ∘ Quotient.eq'.1, fun h ↦ by rw [h]⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


instance : CharZero ℚ_[p] :=
  ⟨fun m n ↦ by
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      m n : Nat
      ⊢ Eq ↑m ↑n → Eq m n
    -/
    rw [← Rat.cast_natCast]
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      m n : Nat
      ⊢ Eq ↑↑m ↑n → Eq m n
    -/
    norm_cast
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      m n : Nat
      ⊢ Eq m n → Eq m n
    -/
    exact id⟩
    /-
      🎉 no goals
    -/


@[norm_cast]
theorem coe_add : ∀ {x y : ℚ}, (↑(x + y) : ℚ_[p]) = ↑x + ↑y :=
  Rat.cast_add _ _


@[norm_cast]
theorem coe_neg : ∀ {x : ℚ}, (↑(-x) : ℚ_[p]) = -↑x :=
  Rat.cast_neg _


@[norm_cast]
theorem coe_mul : ∀ {x y : ℚ}, (↑(x * y) : ℚ_[p]) = ↑x * ↑y :=
  Rat.cast_mul _ _


@[norm_cast]
theorem coe_sub : ∀ {x y : ℚ}, (↑(x - y) : ℚ_[p]) = ↑x - ↑y :=
  Rat.cast_sub _ _


@[norm_cast]
theorem coe_div : ∀ {x y : ℚ}, (↑(x / y) : ℚ_[p]) = ↑x / ↑y :=
  Rat.cast_div _ _


@[norm_cast]
theorem coe_one : (↑(1 : ℚ) : ℚ_[p]) = 1 := rfl


@[norm_cast]
theorem coe_zero : (↑(0 : ℚ) : ℚ_[p]) = 0 := rfl


/-- The rational-valued `p`-adic norm on `ℚ_[p]` is lifted from the norm on Cauchy sequences. The
canonical form of this function is the normed space instance, with notation `‖ ‖`. -/
def padicNormE {p : ℕ} [hp : Fact p.Prime] : AbsoluteValue ℚ_[p] ℚ where
  toFun := Quotient.lift PadicSeq.norm <| @PadicSeq.norm_equiv _ _
  map_mul' q r := Quotient.inductionOn₂ q r <| PadicSeq.norm_mul
  nonneg' q := Quotient.inductionOn q <| PadicSeq.norm_nonneg
  eq_zero' q := Quotient.inductionOn q fun r ↦ by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Padic p
      r : CauSeq Rat (padicNorm p)
      ⊢ Iff (Eq ({ toFun := Quotient.lift PadicSeq.norm ⋯, map_mul' := ⋯ }.toFun (Qu …
    -/
    rw [Padic.zero_def, Quotient.eq]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q : Padic p
      r : CauSeq Rat (padicNorm p)
      ⊢ Iff (Eq ({ toFun := Quotient.lift PadicSeq.norm ⋯, map_mul' := ⋯ }.toFun (Qu …
    -/
    exact PadicSeq.norm_zero_iff r
    /-
      🎉 no goals
    -/
  add_le' q r := by
    trans
      max ((Quotient.lift PadicSeq.norm <| @PadicSeq.norm_equiv _ _) q)
        ((Quotient.lift PadicSeq.norm <| @PadicSeq.norm_equiv _ _) r)
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        q r : Padic p
        ⊢ LE.le ({ toFun := Quotient.lift PadicSeq.norm ⋯, map_mul' := ⋯ }.toFun (HAdd …
      -/
    · exact Quotient.inductionOn₂ q r <| PadicSeq.norm_nonarchimedean
      /-
        🎉 no goals
      -/
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q r : Padic p
      ⊢ LE.le (Max.max (Quotient.lift PadicSeq.norm ⋯ q) (Quotient.lift PadicSeq.nor …
    -/
    refine max_le_add_of_nonneg (Quotient.inductionOn q <| PadicSeq.norm_nonneg) ?_
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      q r : Padic p
      ⊢ LE.le 0 (Quotient.lift PadicSeq.norm ⋯ r)
    -/
    exact Quotient.inductionOn r <| PadicSeq.norm_nonneg
    /-
      🎉 no goals
    -/


theorem defn (f : PadicSeq p) {ε : ℚ} (hε : 0 < ε) :
    ∃ N, ∀ i ≥ N, padicNormE (Padic.mk f - f i : ℚ_[p]) < ε := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    ε : Rat
    hε : LT.lt 0 ε
    ⊢ Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (Padic …
  -/
  dsimp [padicNormE]
  -- `change ∃ N, ∀ i ≥ N, (f - const _ (f i)).norm < ε` also works, but is very slow
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    ε : Rat
    hε : LT.lt 0 ε
    ⊢ Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt (Quotient.lift PadicSeq.norm  …
  -/
  suffices hyp : ∃ N, ∀ i ≥ N, (f - const _ (f i)).norm < ε by peel hyp with N; use N
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    ε : Rat
    hε : LT.lt 0 ε
    ⊢ Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt (HSub.hSub f (CauSeq.const (p …
  -/
  by_contra! h
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    ε : Rat
    hε : LT.lt 0 ε
    h : ∀ (N : Nat), Exists fun i => And (GE.ge i N) (LE.le ε (HSub.hSub f (CauSeq …
    ⊢ False
  -/
  cases' cauchy₂ f hε with N hN
  /-
    case intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    ε : Rat
    hε : LT.lt 0 ε
    h : ∀ (N : Nat), Exists fun i => And (GE.ge i N) (LE.le ε (HSub.hSub f (CauSeq …
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (padicNorm p (HSu …
    ⊢ False
  -/
  rcases h N with ⟨i, hi, hge⟩
  have hne : ¬f - const (padicNorm p) (f i) ≈ 0 := fun h ↦ by
    rw [PadicSeq.norm, dif_pos h] at hge
    exact not_lt_of_ge hge hε
  /-
    case intro.intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    ε : Rat
    hε : LT.lt 0 ε
    h : ∀ (N : Nat), Exists fun i => And (GE.ge i N) (LE.le ε (HSub.hSub f (CauSeq …
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (padicNorm p (HSu …
    i : Nat
    hi : GE.ge i N
    hge : LE.le ε (HSub.hSub f (CauSeq.const (padicNorm p) (↑f i))).norm
    hne : Not (HasEquiv.Equiv (HSub.hSub f (CauSeq.const (padicNorm p) (↑f i))) 0)
    ⊢ False
  -/
  unfold PadicSeq.norm at hge; split_ifs at hge
  /-
    case intro.intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : PadicSeq p
    ε : Rat
    hε : LT.lt 0 ε
    h : ∀ (N : Nat), Exists fun i => And (GE.ge i N) (LE.le ε (HSub.hSub f (CauSeq …
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (padicNorm p (HSu …
    i : Nat
    hi : GE.ge i N
    hne : Not (HasEquiv.Equiv (HSub.hSub f (CauSeq.const (padicNorm p) (↑f i))) 0)
    hge : LE.le ε (padicNorm p (↑(HSub.hSub f (CauSeq.const (padicNorm p) (↑f i))) …
    ⊢ False
  -/
  apply not_le_of_gt _ hge
  cases _root_.le_total N (stationaryPoint hne) with
  | inl hgen =>
    exact hN _ hgen _ hi
  | inr hngen =>
    have := stationaryPoint_spec hne le_rfl hngen
    rw [← this]
    exact hN _ le_rfl _ hi


/-- Theorems about `padicNormE` are named with a `'` so the names do not conflict with the
equivalent theorems about `norm` (`‖ ‖`). -/
theorem nonarchimedean' (q r : ℚ_[p]) :
    padicNormE (q + r : ℚ_[p]) ≤ max (padicNormE q) (padicNormE r) :=
  Quotient.inductionOn₂ q r <| norm_nonarchimedean


/-- Theorems about `padicNormE` are named with a `'` so the names do not conflict with the
equivalent theorems about `norm` (`‖ ‖`). -/
theorem add_eq_max_of_ne' {q r : ℚ_[p]} :
    padicNormE q ≠ padicNormE r → padicNormE (q + r : ℚ_[p]) = max (padicNormE q) (padicNormE r) :=
  Quotient.inductionOn₂ q r fun _ _ ↦ PadicSeq.add_eq_max_of_ne


@[simp]
theorem eq_padic_norm' (q : ℚ) : padicNormE (q : ℚ_[p]) = padicNorm p q :=
  norm_const _


protected theorem image' {q : ℚ_[p]} : q ≠ 0 → ∃ n : ℤ, padicNormE q = (p : ℚ) ^ (-n) :=
  Quotient.inductionOn q fun f hf ↦
    have : ¬f ≈ 0 := (ne_zero_iff_nequiv_zero f).1 hf
    norm_values_discrete f this


theorem rat_dense' (q : ℚ_[p]) {ε : ℚ} (hε : 0 < ε) : ∃ r : ℚ, padicNormE (q - r : ℚ_[p]) < ε :=
  Quotient.inductionOn q fun q' ↦
    have : ∃ N, ∀ m ≥ N, ∀ n ≥ N, padicNorm p (q' m - q' n) < ε := cauchy₂ _ hε
    let ⟨N, hN⟩ := this
    ⟨q' N, by
      classical
      dsimp [padicNormE]
      -- Porting note: `change` → `convert_to` (`change` times out!)
      -- and add `PadicSeq p` type annotation
      convert_to PadicSeq.norm (q' - const _ (q' N) : PadicSeq p) < ε
      cases' Decidable.em (q' - const (padicNorm p) (q' N) ≈ 0) with heq hne'
      · simpa only [heq, PadicSeq.norm, dif_pos]
      · simp only [PadicSeq.norm, dif_neg hne']
        change padicNorm p (q' _ - q' _) < ε
        cases' Decidable.em (stationaryPoint hne' ≤ N) with hle hle
        · -- Porting note: inlined `stationaryPoint_spec` invocation.
          have := (stationaryPoint_spec hne' le_rfl hle).symm
          simp only [const_apply, sub_apply, padicNorm.zero, sub_self] at this
          simpa only [this]
        · exact hN _ (lt_of_not_ge hle).le _ le_rfl⟩


private theorem div_nat_pos (n : ℕ) : 0 < 1 / (n + 1 : ℚ) :=
  div_pos zero_lt_one (mod_cast succ_pos _)


/-- `limSeq f`, for `f` a Cauchy sequence of `p`-adic numbers, is a sequence of rationals with the
same limit point as `f`. -/
def limSeq : ℕ → ℚ :=
  fun n ↦ Classical.choose (rat_dense' (f n) (div_nat_pos n))


theorem exi_rat_seq_conv {ε : ℚ} (hε : 0 < ε) :
    ∃ N, ∀ i ≥ N, padicNormE (f i - (limSeq f i : ℚ_[p]) : ℚ_[p]) < ε := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ε : Rat
    hε : LT.lt 0 ε
    ⊢ Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) …
  -/
  refine (exists_nat_gt (1 / ε)).imp fun N hN i hi ↦ ?_
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ε : Rat
    hε : LT.lt 0 ε
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 ε) ↑N
    i : Nat
    hi : GE.ge i N
    ⊢ LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limSeq f i))) ε
  -/
  have h := Classical.choose_spec (rat_dense' (f i) (div_nat_pos i))
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ε : Rat
    hε : LT.lt 0 ε
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 ε) ↑N
    i : Nat
    hi : GE.ge i N
    h : LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Classical.choose ⋯))) (HDiv.hDiv 1 ( …
    ⊢ LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limSeq f i))) ε
  -/
  refine lt_of_lt_of_le h ((div_le_iff₀' <| mod_cast succ_pos _).mpr ?_)
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ε : Rat
    hε : LT.lt 0 ε
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 ε) ↑N
    i : Nat
    hi : GE.ge i N
    h : LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Classical.choose ⋯))) (HDiv.hDiv 1 ( …
    ⊢ LE.le 1 (HMul.hMul (HAdd.hAdd (↑i) 1) ε)
  -/
  rw [right_distrib]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ε : Rat
    hε : LT.lt 0 ε
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 ε) ↑N
    i : Nat
    hi : GE.ge i N
    h : LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Classical.choose ⋯))) (HDiv.hDiv 1 ( …
    ⊢ LE.le 1 (HAdd.hAdd (HMul.hMul (↑i) ε) (HMul.hMul 1 ε))
  -/
  apply le_add_of_le_of_nonneg
    /-
      case hbc
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : LT.lt 0 ε
      N : Nat
      hN : LT.lt (HDiv.hDiv 1 ε) ↑N
      i : Nat
      hi : GE.ge i N
      h : LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Classical.choose ⋯))) (HDiv.hDiv 1 ( …
      ⊢ LE.le 1 (HMul.hMul (↑i) ε)
    -/
  · exact (div_le_iff₀ hε).mp (le_trans (le_of_lt hN) (mod_cast hi))
    /-
      🎉 no goals
    -/
    /-
      case ha
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : LT.lt 0 ε
      N : Nat
      hN : LT.lt (HDiv.hDiv 1 ε) ↑N
      i : Nat
      hi : GE.ge i N
      h : LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Classical.choose ⋯))) (HDiv.hDiv 1 ( …
      ⊢ LE.le 0 (HMul.hMul 1 ε)
    -/
  · apply le_of_lt
    /-
      case ha.hab
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : LT.lt 0 ε
      N : Nat
      hN : LT.lt (HDiv.hDiv 1 ε) ↑N
      i : Nat
      hi : GE.ge i N
      h : LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Classical.choose ⋯))) (HDiv.hDiv 1 ( …
      ⊢ LT.lt 0 (HMul.hMul 1 ε)
    -/
    simpa
    /-
      🎉 no goals
    -/


theorem exi_rat_seq_conv_cauchy : IsCauSeq (padicNorm p) (limSeq f) := fun ε hε ↦ by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ε : Rat
    hε : GT.gt ε 0
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (HSub.hSub (Padi …
  -/
  have hε3 : 0 < ε / 3 := div_pos hε (by norm_num)
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ε : Rat
    hε : GT.gt ε 0
    hε3 : LT.lt 0 (HDiv.hDiv ε 3)
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (HSub.hSub (Padi …
  -/
  let ⟨N, hN⟩ := exi_rat_seq_conv f hε3
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ε : Rat
    hε : GT.gt ε 0
    hε3 : LT.lt 0 (HDiv.hDiv ε 3)
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (HSub.hSub (Padi …
  -/
  let ⟨N2, hN2⟩ := f.cauchy₂ hε3
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ε : Rat
    hε : GT.gt ε 0
    hε3 : LT.lt 0 (HDiv.hDiv ε 3)
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (HSub.hSub (Padi …
  -/
  exists max N N2
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ε : Rat
    hε : GT.gt ε 0
    hε3 : LT.lt 0 (HDiv.hDiv ε 3)
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
    ⊢ ∀ (j : Nat), GE.ge j (Max.max N N2) → LT.lt (padicNorm p (HSub.hSub (Padic.l …
  -/
  intro j hj
  suffices
    padicNormE (limSeq f j - f (max N N2) + (f (max N N2) - limSeq f (max N N2)) : ℚ_[p]) < ε by
    ring_nf at this ⊢
    rw [← padicNormE.eq_padic_norm']
    exact mod_cast this
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ε : Rat
    hε : GT.gt ε 0
    hε3 : LT.lt 0 (HDiv.hDiv ε 3)
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
    N2 : Nat
    hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
    j : Nat
    hj : GE.ge j (Max.max N N2)
    ⊢ LT.lt (padicNormE (HAdd.hAdd (HSub.hSub (↑(Padic.limSeq f j)) (↑f (Max.max N …
  -/
  apply lt_of_le_of_lt
    /-
      case hab
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : GT.gt ε 0
      hε3 : LT.lt 0 (HDiv.hDiv ε 3)
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
      N2 : Nat
      hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
      j : Nat
      hj : GE.ge j (Max.max N N2)
      ⊢ LE.le (padicNormE (HAdd.hAdd (HSub.hSub (↑(Padic.limSeq f j)) (↑f (Max.max N …
    -/
  · apply padicNormE.add_le
    /-
      🎉 no goals
    -/
    /-
      case hbc
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : GT.gt ε 0
      hε3 : LT.lt 0 (HDiv.hDiv ε 3)
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
      N2 : Nat
      hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
      j : Nat
      hj : GE.ge j (Max.max N N2)
      ⊢ LT.lt (HAdd.hAdd (padicNormE (HSub.hSub (↑(Padic.limSeq f j)) (↑f (Max.max N …
    -/
  · rw [← add_thirds ε]
    /-
      case hbc
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : GT.gt ε 0
      hε3 : LT.lt 0 (HDiv.hDiv ε 3)
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
      N2 : Nat
      hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
      j : Nat
      hj : GE.ge j (Max.max N N2)
      ⊢ LT.lt (HAdd.hAdd (padicNormE (HSub.hSub (↑(Padic.limSeq f j)) (↑f (Max.max N …
    -/
    apply _root_.add_lt_add
    · suffices padicNormE (limSeq f j - f j + (f j - f (max N N2)) : ℚ_[p]) < ε / 3 + ε / 3 by
        simpa only [sub_add_sub_cancel]
      /-
        case hbc.h₁
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        f : CauSeq (Padic p) ⇑padicNormE
        ε : Rat
        hε : GT.gt ε 0
        hε3 : LT.lt 0 (HDiv.hDiv ε 3)
        N : Nat
        hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
        N2 : Nat
        hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
        j : Nat
        hj : GE.ge j (Max.max N N2)
        ⊢ LT.lt (padicNormE (HAdd.hAdd (HSub.hSub (↑(Padic.limSeq f j)) (↑f j)) (HSub. …
      -/
      apply lt_of_le_of_lt
        /-
          case hbc.h₁.hab
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          f : CauSeq (Padic p) ⇑padicNormE
          ε : Rat
          hε : GT.gt ε 0
          hε3 : LT.lt 0 (HDiv.hDiv ε 3)
          N : Nat
          hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
          N2 : Nat
          hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
          j : Nat
          hj : GE.ge j (Max.max N N2)
          ⊢ LE.le (padicNormE (HAdd.hAdd (HSub.hSub (↑(Padic.limSeq f j)) (↑f j)) (HSub. …
        -/
      · apply padicNormE.add_le
        /-
          🎉 no goals
        -/
        /-
          case hbc.h₁.hbc
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          f : CauSeq (Padic p) ⇑padicNormE
          ε : Rat
          hε : GT.gt ε 0
          hε3 : LT.lt 0 (HDiv.hDiv ε 3)
          N : Nat
          hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
          N2 : Nat
          hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
          j : Nat
          hj : GE.ge j (Max.max N N2)
          ⊢ LT.lt (HAdd.hAdd (padicNormE (HSub.hSub (↑(Padic.limSeq f j)) (↑f j))) (padi …
        -/
      · apply _root_.add_lt_add
          /-
            case hbc.h₁.hbc.h₁
            p : Nat
            inst✝ : Fact (Nat.Prime p)
            f : CauSeq (Padic p) ⇑padicNormE
            ε : Rat
            hε : GT.gt ε 0
            hε3 : LT.lt 0 (HDiv.hDiv ε 3)
            N : Nat
            hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
            N2 : Nat
            hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
            j : Nat
            hj : GE.ge j (Max.max N N2)
            ⊢ LT.lt (padicNormE (HSub.hSub (↑(Padic.limSeq f j)) (↑f j))) (HDiv.hDiv ε 3)
          -/
        · rw [padicNormE.map_sub]
          /-
            case hbc.h₁.hbc.h₁
            p : Nat
            inst✝ : Fact (Nat.Prime p)
            f : CauSeq (Padic p) ⇑padicNormE
            ε : Rat
            hε : GT.gt ε 0
            hε3 : LT.lt 0 (HDiv.hDiv ε 3)
            N : Nat
            hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
            N2 : Nat
            hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
            j : Nat
            hj : GE.ge j (Max.max N N2)
            ⊢ LT.lt (padicNormE (HSub.hSub (↑f j) ↑(Padic.limSeq f j))) (HDiv.hDiv ε 3)
          -/
          apply mod_cast hN j
          /-
            case hbc.h₁.hbc.h₁
            p : Nat
            inst✝ : Fact (Nat.Prime p)
            f : CauSeq (Padic p) ⇑padicNormE
            ε : Rat
            hε : GT.gt ε 0
            hε3 : LT.lt 0 (HDiv.hDiv ε 3)
            N : Nat
            hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
            N2 : Nat
            hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
            j : Nat
            hj : GE.ge j (Max.max N N2)
            ⊢ GE.ge j N
          -/
          exact le_of_max_le_left hj
          /-
            🎉 no goals
          -/
          /-
            case hbc.h₁.hbc.h₂
            p : Nat
            inst✝ : Fact (Nat.Prime p)
            f : CauSeq (Padic p) ⇑padicNormE
            ε : Rat
            hε : GT.gt ε 0
            hε3 : LT.lt 0 (HDiv.hDiv ε 3)
            N : Nat
            hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
            N2 : Nat
            hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
            j : Nat
            hj : GE.ge j (Max.max N N2)
            ⊢ LT.lt (padicNormE (HSub.hSub (↑f j) (↑f (Max.max N N2)))) (HDiv.hDiv ε 3)
          -/
        · exact hN2 _ (le_of_max_le_right hj) _ (le_max_right _ _)
          /-
            🎉 no goals
          -/
      /-
        case hbc.h₂
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        f : CauSeq (Padic p) ⇑padicNormE
        ε : Rat
        hε : GT.gt ε 0
        hε3 : LT.lt 0 (HDiv.hDiv ε 3)
        N : Nat
        hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
        N2 : Nat
        hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
        j : Nat
        hj : GE.ge j (Max.max N N2)
        ⊢ LT.lt (padicNormE (HSub.hSub (↑f (Max.max N N2)) ↑(Padic.limSeq f (Max.max N …
      -/
    · apply mod_cast hN (max N N2)
      /-
        case hbc.h₂
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        f : CauSeq (Padic p) ⇑padicNormE
        ε : Rat
        hε : GT.gt ε 0
        hε3 : LT.lt 0 (HDiv.hDiv ε 3)
        N : Nat
        hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
        N2 : Nat
        hN2 : ∀ (j : Nat), GE.ge j N2 → ∀ (k : Nat), GE.ge k N2 → LT.lt (padicNormE (H …
        j : Nat
        hj : GE.ge j (Max.max N N2)
        ⊢ GE.ge (Max.max N N2) N
      -/
      apply le_max_left
      /-
        🎉 no goals
      -/


private def lim' : PadicSeq p :=
  ⟨_, exi_rat_seq_conv_cauchy f⟩


private def lim : ℚ_[p] :=
  ⟦lim' f⟧


theorem complete' : ∃ q : ℚ_[p], ∀ ε > 0, ∃ N, ∀ i ≥ N, padicNormE (q - f i : ℚ_[p]) < ε :=
  ⟨lim f, fun ε hε ↦ by
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : GT.gt ε 0
      ⊢ Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (Padic …
    -/
    obtain ⟨N, hN⟩ := exi_rat_seq_conv f (half_pos hε)
    /-
      case intro
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : GT.gt ε 0
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
      ⊢ Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (Padic …
    -/
    obtain ⟨N2, hN2⟩ := padicNormE.defn (lim' f) (half_pos hε)
    /-
      case intro.intro
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : GT.gt ε 0
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
      N2 : Nat
      hN2 : ∀ (i : Nat), GE.ge i N2 → LT.lt (padicNormE (HSub.hSub (Padic.mk (Padic. …
      ⊢ Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (Padic …
    -/
    refine ⟨max N N2, fun i hi ↦ ?_⟩
    /-
      case intro.intro
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : GT.gt ε 0
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
      N2 : Nat
      hN2 : ∀ (i : Nat), GE.ge i N2 → LT.lt (padicNormE (HSub.hSub (Padic.mk (Padic. …
      i : Nat
      hi : GE.ge i (Max.max N N2)
      ⊢ LT.lt (padicNormE (HSub.hSub (Padic.lim f) (↑f i))) ε
    -/
    rw [← sub_add_sub_cancel _ (lim' f i : ℚ_[p]) _]
    /-
      case intro.intro
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : GT.gt ε 0
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
      N2 : Nat
      hN2 : ∀ (i : Nat), GE.ge i N2 → LT.lt (padicNormE (HSub.hSub (Padic.mk (Padic. …
      i : Nat
      hi : GE.ge i (Max.max N N2)
      ⊢ LT.lt (padicNormE (HAdd.hAdd (HSub.hSub (Padic.lim f) ↑(↑(Padic.lim' f) i))  …
    -/
    refine (padicNormE.add_le _ _).trans_lt ?_
    /-
      case intro.intro
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : GT.gt ε 0
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
      N2 : Nat
      hN2 : ∀ (i : Nat), GE.ge i N2 → LT.lt (padicNormE (HSub.hSub (Padic.mk (Padic. …
      i : Nat
      hi : GE.ge i (Max.max N N2)
      ⊢ LT.lt (HAdd.hAdd (padicNormE (HSub.hSub (Padic.lim f) ↑(↑(Padic.lim' f) i))) …
    -/
    rw [← add_halves ε]
    /-
      case intro.intro
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      f : CauSeq (Padic p) ⇑padicNormE
      ε : Rat
      hε : GT.gt ε 0
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
      N2 : Nat
      hN2 : ∀ (i : Nat), GE.ge i N2 → LT.lt (padicNormE (HSub.hSub (Padic.mk (Padic. …
      i : Nat
      hi : GE.ge i (Max.max N N2)
      ⊢ LT.lt (HAdd.hAdd (padicNormE (HSub.hSub (Padic.lim f) ↑(↑(Padic.lim' f) i))) …
    -/
    apply _root_.add_lt_add
      /-
        case intro.intro.h₁
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        f : CauSeq (Padic p) ⇑padicNormE
        ε : Rat
        hε : GT.gt ε 0
        N : Nat
        hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
        N2 : Nat
        hN2 : ∀ (i : Nat), GE.ge i N2 → LT.lt (padicNormE (HSub.hSub (Padic.mk (Padic. …
        i : Nat
        hi : GE.ge i (Max.max N N2)
        ⊢ LT.lt (padicNormE (HSub.hSub (Padic.lim f) ↑(↑(Padic.lim' f) i))) (HDiv.hDiv …
      -/
    · apply hN2 _ (le_of_max_le_right hi)
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.h₂
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        f : CauSeq (Padic p) ⇑padicNormE
        ε : Rat
        hε : GT.gt ε 0
        N : Nat
        hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
        N2 : Nat
        hN2 : ∀ (i : Nat), GE.ge i N2 → LT.lt (padicNormE (HSub.hSub (Padic.mk (Padic. …
        i : Nat
        hi : GE.ge i (Max.max N N2)
        ⊢ LT.lt (padicNormE (HSub.hSub (↑(↑(Padic.lim' f) i)) (↑f i))) (HDiv.hDiv ε 2)
      -/
    · rw [padicNormE.map_sub]
      /-
        case intro.intro.h₂
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        f : CauSeq (Padic p) ⇑padicNormE
        ε : Rat
        hε : GT.gt ε 0
        N : Nat
        hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) ↑(Padic.limS …
        N2 : Nat
        hN2 : ∀ (i : Nat), GE.ge i N2 → LT.lt (padicNormE (HSub.hSub (Padic.mk (Padic. …
        i : Nat
        hi : GE.ge i (Max.max N N2)
        ⊢ LT.lt (padicNormE (HSub.hSub (↑f i) ↑(↑(Padic.lim' f) i))) (HDiv.hDiv ε 2)
      -/
      exact hN _ (le_of_max_le_left hi)⟩
      /-
        🎉 no goals
      -/


theorem complete'' : ∃ q : ℚ_[p], ∀ ε > 0, ∃ N, ∀ i ≥ N, padicNormE (f i - q : ℚ_[p]) < ε := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    ⊢ Exists fun q => ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge  …
  -/
  obtain ⟨x, hx⟩ := complete' f
  /-
    case intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    x : Padic p
    hx : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
    ⊢ Exists fun q => ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge  …
  -/
  refine ⟨x, fun ε hε => ?_⟩
  /-
    case intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    x : Padic p
    hx : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
    ε : Rat
    hε : GT.gt ε 0
    ⊢ Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) …
  -/
  obtain ⟨N, hN⟩ := hx ε hε
  /-
    case intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    x : Padic p
    hx : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
    ε : Rat
    hε : GT.gt ε 0
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub x (↑f i))) ε
    ⊢ Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑f i) …
  -/
  refine ⟨N, fun i hi => ?_⟩
  /-
    case intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    x : Padic p
    hx : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
    ε : Rat
    hε : GT.gt ε 0
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub x (↑f i))) ε
    i : Nat
    hi : GE.ge i N
    ⊢ LT.lt (padicNormE (HSub.hSub (↑f i) x)) ε
  -/
  rw [padicNormE.map_sub]
  /-
    case intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f : CauSeq (Padic p) ⇑padicNormE
    x : Padic p
    hx : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
    ε : Rat
    hε : GT.gt ε 0
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub x (↑f i))) ε
    i : Nat
    hi : GE.ge i N
    ⊢ LT.lt (padicNormE (HSub.hSub x (↑f i))) ε
  -/
  exact hN i hi
  /-
    🎉 no goals
  -/

instance : Dist ℚ_[p] :=
  ⟨fun x y ↦ padicNormE (x - y : ℚ_[p])⟩


instance : IsUltrametricDist ℚ_[p] :=
                  /-
                    p : Nat
                    inst✝ : Fact (Nat.Prime p)
                    x y z : Padic p
                    ⊢ LE.le (Dist.dist x z) (Max.max (Dist.dist x y) (Dist.dist y z))
                  -/
  ⟨fun x y z ↦ by simpa [dist] using padicNormE.nonarchimedean' (x - y) (y - z)⟩
                  /-
                    🎉 no goals
                  -/


instance metricSpace : MetricSpace ℚ_[p] where
                  /-
                    p : Nat
                    inst✝ : Fact (Nat.Prime p)
                    ⊢ ∀ (x : Padic p), Eq (Dist.dist x x) 0
                  -/
  dist_self := by simp [dist]
                  /-
                    🎉 no goals
                  -/
  dist := dist
                      /-
                        p : Nat
                        inst✝ : Fact (Nat.Prime p)
                        x y : Padic p
                        ⊢ Eq (Dist.dist x y) (Dist.dist y x)
                      -/
  dist_comm x y := by simp [dist, ← padicNormE.map_neg (x - y : ℚ_[p])]
                      /-
                        🎉 no goals
                      -/
  dist_triangle x y z := by
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      x y z : Padic p
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
    dsimp [dist]
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      x y z : Padic p
      ⊢ LE.le (↑(padicNormE (HSub.hSub x z))) (HAdd.hAdd ↑(padicNormE (HSub.hSub x y …
    -/
    exact mod_cast padicNormE.sub_le x y z
    /-
      🎉 no goals
    -/
  eq_of_dist_eq_zero := by
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      ⊢ ∀ {x y : Padic p}, Eq (Dist.dist x y) 0 → Eq x y
    -/
    dsimp [dist]; intro _ _ h
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      x✝ y✝ : Padic p
      h : Eq (↑(padicNormE (HSub.hSub x✝ y✝))) 0
      ⊢ Eq x✝ y✝
    -/
    apply eq_of_sub_eq_zero
    /-
      case h
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      x✝ y✝ : Padic p
      h : Eq (↑(padicNormE (HSub.hSub x✝ y✝))) 0
      ⊢ Eq (HSub.hSub x✝ y✝) 0
    -/
    apply padicNormE.eq_zero.1
    /-
      case h
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      x✝ y✝ : Padic p
      h : Eq (↑(padicNormE (HSub.hSub x✝ y✝))) 0
      ⊢ Eq (padicNormE (HSub.hSub x✝ y✝)) 0
    -/
    exact mod_cast h
    /-
      🎉 no goals
    -/


instance : Norm ℚ_[p] :=
  ⟨fun x ↦ padicNormE x⟩


instance normedField : NormedField ℚ_[p] :=
  { Padic.field,
    Padic.metricSpace p with
    dist_eq := fun _ _ ↦ rfl
                    /-
                      p : Nat
                      inst✝ : Fact (Nat.Prime p)
                      ⊢ ∀ (a b : Padic p), Eq (Norm.norm (HMul.hMul a b)) (HMul.hMul (Norm.norm a) ( …
                    -/
    norm_mul' := by simp [Norm.norm, map_mul]
                    /-
                      🎉 no goals
                    -/
    norm := norm }


instance isAbsoluteValue : IsAbsoluteValue fun a : ℚ_[p] ↦ ‖a‖ where
  abv_nonneg' := norm_nonneg
  abv_eq_zero' := norm_eq_zero
  abv_add' := norm_add_le
                 /-
                   p : Nat
                   inst✝ : Fact (Nat.Prime p)
                   ⊢ ∀ (x y : Padic p), Eq (Norm.norm (HMul.hMul x y)) (HMul.hMul (Norm.norm x) ( …
                 -/
  abv_mul' := by simp [Norm.norm, map_mul]
                 /-
                   🎉 no goals
                 -/


theorem rat_dense (q : ℚ_[p]) {ε : ℝ} (hε : 0 < ε) : ∃ r : ℚ, ‖q - r‖ < ε :=
  let ⟨ε', hε'l, hε'r⟩ := exists_rat_btwn hε
                                            /-
                                              p : Nat
                                              inst✝ : Fact (Nat.Prime p)
                                              q : Padic p
                                              ε : Real
                                              hε : LT.lt 0 ε
                                              ε' : Rat
                                              hε'l : LT.lt 0 ↑ε'
                                              hε'r : LT.lt (↑ε') ε
                                              ⊢ LT.lt 0 ε'
                                            -/
  let ⟨r, hr⟩ := rat_dense' q (ε := ε') (by simpa using hε'l)
                                            /-
                                              🎉 no goals
                                            -/
                   /-
                     p : Nat
                     inst✝ : Fact (Nat.Prime p)
                     q : Padic p
                     ε : Real
                     hε : LT.lt 0 ε
                     ε' : Rat
                     hε'l : LT.lt 0 ↑ε'
                     hε'r : LT.lt (↑ε') ε
                     r : Rat
                     hr : LT.lt (padicNormE (HSub.hSub q ↑r)) ε'
                     ⊢ LT.lt (Norm.norm (HSub.hSub q ↑r)) ↑ε'
                   -/
  ⟨r, lt_trans (by simpa [Norm.norm] using hr) hε'r⟩
                   /-
                     🎉 no goals
                   -/


@[simp (high)]
                                                                /-
                                                                  p : Nat
                                                                  hp : Fact (Nat.Prime p)
                                                                  q r : Padic p
                                                                  ⊢ Eq (Norm.norm (HMul.hMul q r)) (HMul.hMul (Norm.norm q) (Norm.norm r))
                                                                -/
protected theorem mul (q r : ℚ_[p]) : ‖q * r‖ = ‖q‖ * ‖r‖ := by simp [Norm.norm, map_mul]
                                                                /-
                                                                  🎉 no goals
                                                                -/


protected theorem is_norm (q : ℚ_[p]) : ↑(padicNormE q) = ‖q‖ := rfl


theorem nonarchimedean (q r : ℚ_[p]) : ‖q + r‖ ≤ max ‖q‖ ‖r‖ := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Padic p
    ⊢ LE.le (Norm.norm (HAdd.hAdd q r)) (Max.max (Norm.norm q) (Norm.norm r))
  -/
  dsimp [norm]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Padic p
    ⊢ LE.le (↑(padicNormE (HAdd.hAdd q r))) (Max.max ↑(padicNormE q) ↑(padicNormE  …
  -/
  exact mod_cast nonarchimedean' _ _
  /-
    🎉 no goals
  -/


theorem add_eq_max_of_ne {q r : ℚ_[p]} (h : ‖q‖ ≠ ‖r‖) : ‖q + r‖ = max ‖q‖ ‖r‖ := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Padic p
    h : Ne (Norm.norm q) (Norm.norm r)
    ⊢ Eq (Norm.norm (HAdd.hAdd q r)) (Max.max (Norm.norm q) (Norm.norm r))
  -/
  dsimp [norm] at h ⊢
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Padic p
    h : Not (Eq ↑(padicNormE q) ↑(padicNormE r))
    ⊢ Eq (↑(padicNormE (HAdd.hAdd q r))) (Max.max ↑(padicNormE q) ↑(padicNormE r))
  -/
  have : padicNormE q ≠ padicNormE r := mod_cast h
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Padic p
    h : Not (Eq ↑(padicNormE q) ↑(padicNormE r))
    this : Ne (padicNormE q) (padicNormE r)
    ⊢ Eq (↑(padicNormE (HAdd.hAdd q r))) (Max.max ↑(padicNormE q) ↑(padicNormE r))
  -/
  exact mod_cast add_eq_max_of_ne' this
  /-
    🎉 no goals
  -/


@[simp]
theorem eq_padicNorm (q : ℚ) : ‖(q : ℚ_[p])‖ = padicNorm p q := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    ⊢ Eq (Norm.norm ↑q) ↑(padicNorm p q)
  -/
  dsimp [norm]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    ⊢ Eq ↑(padicNormE ↑q) ↑(padicNorm p q)
  -/
  rw [← padicNormE.eq_padic_norm']
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_p : ‖(p : ℚ_[p])‖ = (p : ℝ)⁻¹ := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (Norm.norm ↑p) (Inv.inv ↑p)
  -/
  rw [← @Rat.cast_natCast ℝ _ p]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (Norm.norm ↑p) (Inv.inv ↑↑p)
  -/
  rw [← @Rat.cast_natCast ℚ_[p] _ p]
  simp [hp.1.ne_zero, hp.1.ne_one, norm, padicNorm, padicValRat, padicValInt, zpow_neg,
    -Rat.cast_natCast]


theorem norm_p_lt_one : ‖(p : ℚ_[p])‖ < 1 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ LT.lt (Norm.norm ↑p) 1
  -/
  rw [norm_p]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ LT.lt (Inv.inv ↑p) 1
  -/
  exact inv_lt_one_of_one_lt₀ <| mod_cast hp.1.one_lt
  /-
    🎉 no goals
  -/

-- Porting note: Linter thinks this is a duplicate simp lemma, so `priority` is assigned

@[simp (high)]
theorem norm_p_zpow (n : ℤ) : ‖(p : ℚ_[p]) ^ n‖ = (p : ℝ) ^ (-n) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Int
    ⊢ Eq (Norm.norm (HPow.hPow (↑p) n)) (HPow.hPow (↑p) (Neg.neg n))
  -/
  rw [norm_zpow, norm_p, zpow_neg, inv_zpow]
  /-
    🎉 no goals
  -/

-- Porting note: Linter thinks this is a duplicate simp lemma, so `priority` is assigned

@[simp (high)]
theorem norm_p_pow (n : ℕ) : ‖(p : ℚ_[p]) ^ n‖ = (p : ℝ) ^ (-n : ℤ) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (Norm.norm (HPow.hPow (↑p) n)) (HPow.hPow (↑p) (Neg.neg ↑n))
  -/
  rw [← norm_p_zpow, zpow_natCast]
  /-
    🎉 no goals
  -/


instance : NontriviallyNormedField ℚ_[p] :=
  { Padic.normedField p with
    non_trivial :=
      ⟨p⁻¹, by
        /-
          p : Nat
          hp : Fact (Nat.Prime p)
          ⊢ LT.lt 1 (Norm.norm (Inv.inv ↑p))
        -/
        rw [norm_inv, norm_p, inv_inv]
        /-
          p : Nat
          hp : Fact (Nat.Prime p)
          ⊢ LT.lt 1 ↑p
        -/
        exact mod_cast hp.1.one_lt⟩ }
        /-
          🎉 no goals
        -/


protected theorem image {q : ℚ_[p]} : q ≠ 0 → ∃ n : ℤ, ‖q‖ = ↑((p : ℚ) ^ (-n)) :=
  Quotient.inductionOn q fun f hf ↦
    have : ¬f ≈ 0 := (PadicSeq.ne_zero_iff_nequiv_zero f).1 hf
    let ⟨n, hn⟩ := PadicSeq.norm_values_discrete f this
           /-
             p : Nat
             hp : Fact (Nat.Prime p)
             q : Padic p
             f : CauSeq Rat (padicNorm p)
             hf : Ne (Quotient.mk CauSeq.equiv f) 0
             this : Not (HasEquiv.Equiv f 0)
             n : Int
             hn : Eq (PadicSeq.norm f) (HPow.hPow (↑p) (Neg.neg n))
             ⊢ Eq (Norm.norm (Quotient.mk CauSeq.equiv f)) ↑(HPow.hPow (↑p) (Neg.neg n))
           -/
    ⟨n, by rw [← hn]; rfl⟩
                      /-
                        🎉 no goals
                      -/


protected theorem is_rat (q : ℚ_[p]) : ∃ q' : ℚ, ‖q‖ = q' := by
  classical
  exact if h : q = 0 then ⟨0, by simp [h]⟩
  else
    let ⟨n, hn⟩ := padicNormE.image h
    ⟨_, hn⟩


/-- `ratNorm q`, for a `p`-adic number `q` is the `p`-adic norm of `q`, as rational number.

The lemma `padicNormE.eq_ratNorm` asserts `‖q‖ = ratNorm q`. -/
def ratNorm (q : ℚ_[p]) : ℚ :=
  Classical.choose (padicNormE.is_rat q)


theorem eq_ratNorm (q : ℚ_[p]) : ‖q‖ = ratNorm q :=
  Classical.choose_spec (padicNormE.is_rat q)


theorem norm_rat_le_one : ∀ {q : ℚ} (_ : ¬p ∣ q.den), ‖(q : ℚ_[p])‖ ≤ 1
  | ⟨n, d, hn, hd⟩ => fun hq : ¬p ∣ d ↦
    if hnz : n = 0 then by
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Int
        d : Nat
        hn : Ne d 0
        hd : n.natAbs.Coprime d
        hq : Not (Dvd.dvd p d)
        hnz : Eq n 0
        ⊢ LE.le (Norm.norm ↑{ num := n, den := d, den_nz := hn, reduced := hd }) 1
      -/
      have : (⟨n, d, hn, hd⟩ : ℚ) = 0 := Rat.zero_iff_num_zero.mpr hnz
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Int
        d : Nat
        hn : Ne d 0
        hd : n.natAbs.Coprime d
        hq : Not (Dvd.dvd p d)
        hnz : Eq n 0
        this : Eq { num := n, den := d, den_nz := hn, reduced := hd } 0
        ⊢ LE.le (Norm.norm ↑{ num := n, den := d, den_nz := hn, reduced := hd }) 1
      -/
      norm_num [this]
      /-
        🎉 no goals
      -/
    else by
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Int
        d : Nat
        hn : Ne d 0
        hd : n.natAbs.Coprime d
        hq : Not (Dvd.dvd p d)
        hnz : Not (Eq n 0)
        ⊢ LE.le (Norm.norm ↑{ num := n, den := d, den_nz := hn, reduced := hd }) 1
      -/
      have hnz' : (⟨n, d, hn, hd⟩ : ℚ) ≠ 0 := mt Rat.zero_iff_num_zero.1 hnz
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Int
        d : Nat
        hn : Ne d 0
        hd : n.natAbs.Coprime d
        hq : Not (Dvd.dvd p d)
        hnz : Not (Eq n 0)
        hnz' : Ne { num := n, den := d, den_nz := hn, reduced := hd } 0
        ⊢ LE.le (Norm.norm ↑{ num := n, den := d, den_nz := hn, reduced := hd }) 1
      -/
      rw [padicNormE.eq_padicNorm]
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Int
        d : Nat
        hn : Ne d 0
        hd : n.natAbs.Coprime d
        hq : Not (Dvd.dvd p d)
        hnz : Not (Eq n 0)
        hnz' : Ne { num := n, den := d, den_nz := hn, reduced := hd } 0
        ⊢ LE.le (↑(padicNorm p { num := n, den := d, den_nz := hn, reduced := hd })) 1
      -/
      norm_cast
      -- Porting note: `Nat.cast_zero` instead of another `norm_cast` call
      rw [padicNorm.eq_zpow_of_nonzero hnz', padicValRat, neg_sub,
        padicValNat.eq_zero_of_not_dvd hq, Nat.cast_zero, zero_sub, zpow_neg, zpow_natCast]
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Int
        d : Nat
        hn : Ne d 0
        hd : n.natAbs.Coprime d
        hq : Not (Dvd.dvd p d)
        hnz : Not (Eq n 0)
        hnz' : Ne { num := n, den := d, den_nz := hn, reduced := hd } 0
        ⊢ LE.le (Inv.inv (HPow.hPow (↑p) (padicValInt p { num := n, den := d, den_nz : …
      -/
      apply inv_le_one_of_one_le₀
      /-
        case ha
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Int
        d : Nat
        hn : Ne d 0
        hd : n.natAbs.Coprime d
        hq : Not (Dvd.dvd p d)
        hnz : Not (Eq n 0)
        hnz' : Ne { num := n, den := d, den_nz := hn, reduced := hd } 0
        ⊢ LE.le 1 (HPow.hPow (↑p) (padicValInt p { num := n, den := d, den_nz := hn, r …
      -/
      norm_cast
      /-
        case ha
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Int
        d : Nat
        hn : Ne d 0
        hd : n.natAbs.Coprime d
        hq : Not (Dvd.dvd p d)
        hnz : Not (Eq n 0)
        hnz' : Ne { num := n, den := d, den_nz := hn, reduced := hd } 0
        ⊢ LE.le 1 (HPow.hPow p (padicValInt p { num := n, den := d, den_nz := hn, redu …
      -/
      apply one_le_pow
      /-
        case ha.h
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Int
        d : Nat
        hn : Ne d 0
        hd : n.natAbs.Coprime d
        hq : Not (Dvd.dvd p d)
        hnz : Not (Eq n 0)
        hnz' : Ne { num := n, den := d, den_nz := hn, reduced := hd } 0
        ⊢ LT.lt 0 p
      -/
      exact hp.1.pos
      /-
        🎉 no goals
      -/


theorem norm_int_le_one (z : ℤ) : ‖(z : ℚ_[p])‖ ≤ 1 :=
                                      /-
                                        p : Nat
                                        hp : Fact (Nat.Prime p)
                                        z : Int
                                        this : LE.le (Norm.norm ↑↑z) 1
                                        ⊢ LE.le (Norm.norm ↑z) 1
                                      -/
                        /-
                          p : Nat
                          hp : Fact (Nat.Prime p)
                          z : Int
                          ⊢ Not (Dvd.dvd p (↑z).den)
                        -/
  suffices ‖((z : ℚ) : ℚ_[p])‖ ≤ 1 by simpa
                        /-
                          🎉 no goals
                        -/
                                      /-
                                        🎉 no goals
                                      -/
  norm_rat_le_one <| by simp [hp.1.ne_one]


theorem norm_int_lt_one_iff_dvd (k : ℤ) : ‖(k : ℚ_[p])‖ < 1 ↔ ↑p ∣ k := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Int
    ⊢ Iff (LT.lt (Norm.norm ↑k) 1) (Dvd.dvd (↑p) k)
  -/
  constructor
    /-
      case mp
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Int
      ⊢ LT.lt (Norm.norm ↑k) 1 → Dvd.dvd (↑p) k
    -/
  · intro h
    /-
      case mp
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Int
      h : LT.lt (Norm.norm ↑k) 1
      ⊢ Dvd.dvd (↑p) k
    -/
    contrapose! h
    /-
      case mp
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Int
      h : Not (Dvd.dvd (↑p) k)
      ⊢ LE.le 1 (Norm.norm ↑k)
    -/
    apply le_of_eq
    /-
      case mp.hab
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Int
      h : Not (Dvd.dvd (↑p) k)
      ⊢ Eq 1 (Norm.norm ↑k)
    -/
    rw [eq_comm]
    calc
      ‖(k : ℚ_[p])‖ = ‖((k : ℚ) : ℚ_[p])‖ := by norm_cast
      _ = padicNorm p k := padicNormE.eq_padicNorm _
      _ = 1 := mod_cast (int_eq_one_iff k).mpr h
    /-
      case mpr
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Int
      ⊢ Dvd.dvd (↑p) k → LT.lt (Norm.norm ↑k) 1
    -/
  · rintro ⟨x, rfl⟩
    /-
      case mpr.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Int
      ⊢ LT.lt (Norm.norm ↑(HMul.hMul (↑p) x)) 1
    -/
    push_cast
    /-
      case mpr.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Int
      ⊢ LT.lt (Norm.norm (HMul.hMul ↑p ↑x)) 1
    -/
    rw [padicNormE.mul]
    calc
      _ ≤ ‖(p : ℚ_[p])‖ * 1 :=
        mul_le_mul le_rfl (by simpa using norm_int_le_one _) (norm_nonneg _) (norm_nonneg _)
      _ < 1 := by
        rw [mul_one, padicNormE.norm_p]
        exact inv_lt_one_of_one_lt₀ <| mod_cast hp.1.one_lt


theorem norm_int_le_pow_iff_dvd (k : ℤ) (n : ℕ) :
    ‖(k : ℚ_[p])‖ ≤ (p : ℝ) ^ (-n : ℤ) ↔ (p ^ n : ℤ) ∣ k := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Int
    n : Nat
    ⊢ Iff (LE.le (Norm.norm ↑k) (HPow.hPow (↑p) (Neg.neg ↑n))) (Dvd.dvd (HPow.hPow …
  -/
  have : (p : ℝ) ^ (-n : ℤ) = (p : ℚ) ^ (-n : ℤ) := by simp
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Int
    n : Nat
    this : Eq (HPow.hPow (↑p) (Neg.neg ↑n)) (HPow.hPow (↑↑p) (Neg.neg ↑n))
    ⊢ Iff (LE.le (Norm.norm ↑k) (HPow.hPow (↑p) (Neg.neg ↑n))) (Dvd.dvd (HPow.hPow …
  -/
  rw [show (k : ℚ_[p]) = ((k : ℚ) : ℚ_[p]) by norm_cast, eq_padicNorm, this]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Int
    n : Nat
    this : Eq (HPow.hPow (↑p) (Neg.neg ↑n)) (HPow.hPow (↑↑p) (Neg.neg ↑n))
    ⊢ Iff (LE.le (↑(padicNorm p ↑k)) (HPow.hPow (↑↑p) (Neg.neg ↑n))) (Dvd.dvd (HPo …
  -/
  norm_cast
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Int
    n : Nat
    this : Eq (HPow.hPow (↑p) (Neg.neg ↑n)) (HPow.hPow (↑↑p) (Neg.neg ↑n))
    ⊢ Iff (LE.le (padicNorm p ↑k) (HPow.hPow (↑p) (Neg.neg ↑n))) (Dvd.dvd (↑(HPow. …
  -/
  rw [← padicNorm.dvd_iff_norm_le]
  /-
    🎉 no goals
  -/


theorem eq_of_norm_add_lt_right {z1 z2 : ℚ_[p]} (h : ‖z1 + z2‖ < ‖z2‖) : ‖z1‖ = ‖z2‖ :=
  _root_.by_contradiction fun hne ↦
                     /-
                       p : Nat
                       hp : Fact (Nat.Prime p)
                       z1 z2 : Padic p
                       h : LT.lt (Norm.norm (HAdd.hAdd z1 z2)) (Norm.norm z2)
                       hne : Not (Eq (Norm.norm z1) (Norm.norm z2))
                       ⊢ GE.ge (Norm.norm (HAdd.hAdd z1 z2)) (Norm.norm z2)
                     -/
    not_lt_of_ge (by rw [padicNormE.add_eq_max_of_ne hne]; apply le_max_right) h
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem eq_of_norm_add_lt_left {z1 z2 : ℚ_[p]} (h : ‖z1 + z2‖ < ‖z1‖) : ‖z1‖ = ‖z2‖ :=
  _root_.by_contradiction fun hne ↦
                     /-
                       p : Nat
                       hp : Fact (Nat.Prime p)
                       z1 z2 : Padic p
                       h : LT.lt (Norm.norm (HAdd.hAdd z1 z2)) (Norm.norm z1)
                       hne : Not (Eq (Norm.norm z1) (Norm.norm z2))
                       ⊢ GE.ge (Norm.norm (HAdd.hAdd z1 z2)) (Norm.norm z1)
                     -/
    not_lt_of_ge (by rw [padicNormE.add_eq_max_of_ne hne]; apply le_max_left) h
                                                           /-
                                                             🎉 no goals
                                                           -/


instance complete : CauSeq.IsComplete ℚ_[p] norm where
  isComplete f := by
    have cau_seq_norm_e : IsCauSeq padicNormE f := fun ε hε => by
      have h := isCauSeq f ε (mod_cast hε)
      dsimp [norm] at h
      exact mod_cast h
    -- Porting note: Padic.complete' works with `f i - q`, but the goal needs `q - f i`,
    -- using `rewrite [padicNormE.map_sub]` causes time out, so a separate lemma is created
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      ⊢ Exists fun b => HasEquiv.Equiv f (CauSeq.const Norm.norm b)
    -/
    cases' Padic.complete'' ⟨f, cau_seq_norm_e⟩ with q hq
    /-
      case intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ⊢ Exists fun b => HasEquiv.Equiv f (CauSeq.const Norm.norm b)
    -/
    exists q
    /-
      case intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ⊢ HasEquiv.Equiv f (CauSeq.const Norm.norm q)
    -/
    intro ε hε
    /-
      case intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ε : Real
      hε : GT.gt ε 0
      ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (↑(HSub.hSub f (Ca …
    -/
    cases' exists_rat_btwn hε with ε' hε'
    /-
      case intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ε : Real
      hε : GT.gt ε 0
      ε' : Rat
      hε' : And (LT.lt 0 ↑ε') (LT.lt (↑ε') ε)
      ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (↑(HSub.hSub f (Ca …
    -/
    norm_cast at hε'
    /-
      case intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ε : Real
      hε : GT.gt ε 0
      ε' : Rat
      hε' : And (LT.lt 0 ε') (LT.lt (↑ε') ε)
      ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (↑(HSub.hSub f (Ca …
    -/
    cases' hq ε' hε'.1 with N hN
    /-
      case intro.intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ε : Real
      hε : GT.gt ε 0
      ε' : Rat
      hε' : And (LT.lt 0 ε') (LT.lt (↑ε') ε)
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑⟨↑f, cau_seq_norm …
      ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (↑(HSub.hSub f (Ca …
    -/
    exists N
    /-
      case intro.intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ε : Real
      hε : GT.gt ε 0
      ε' : Rat
      hε' : And (LT.lt 0 ε') (LT.lt (↑ε') ε)
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑⟨↑f, cau_seq_norm …
      ⊢ ∀ (j : Nat), GE.ge j N → LT.lt (Norm.norm (↑(HSub.hSub f (CauSeq.const Norm. …
    -/
    intro i hi
    /-
      case intro.intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ε : Real
      hε : GT.gt ε 0
      ε' : Rat
      hε' : And (LT.lt 0 ε') (LT.lt (↑ε') ε)
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑⟨↑f, cau_seq_norm …
      i : Nat
      hi : GE.ge i N
      ⊢ LT.lt (Norm.norm (↑(HSub.hSub f (CauSeq.const Norm.norm q)) i)) ε
    -/
    have h := hN i hi
    /-
      case intro.intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ε : Real
      hε : GT.gt ε 0
      ε' : Rat
      hε' : And (LT.lt 0 ε') (LT.lt (↑ε') ε)
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑⟨↑f, cau_seq_norm …
      i : Nat
      hi : GE.ge i N
      h : LT.lt (padicNormE (HSub.hSub (↑⟨↑f, cau_seq_norm_e⟩ i) q)) ε'
      ⊢ LT.lt (Norm.norm (↑(HSub.hSub f (CauSeq.const Norm.norm q)) i)) ε
    -/
    change norm (f i - q) < ε
    /-
      case intro.intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ε : Real
      hε : GT.gt ε 0
      ε' : Rat
      hε' : And (LT.lt 0 ε') (LT.lt (↑ε') ε)
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑⟨↑f, cau_seq_norm …
      i : Nat
      hi : GE.ge i N
      h : LT.lt (padicNormE (HSub.hSub (↑⟨↑f, cau_seq_norm_e⟩ i) q)) ε'
      ⊢ LT.lt (Norm.norm (HSub.hSub (↑f i) q)) ε
    -/
    refine lt_trans ?_ hε'.2
    /-
      case intro.intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ε : Real
      hε : GT.gt ε 0
      ε' : Rat
      hε' : And (LT.lt 0 ε') (LT.lt (↑ε') ε)
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑⟨↑f, cau_seq_norm …
      i : Nat
      hi : GE.ge i N
      h : LT.lt (padicNormE (HSub.hSub (↑⟨↑f, cau_seq_norm_e⟩ i) q)) ε'
      ⊢ LT.lt (Norm.norm (HSub.hSub (↑f i) q)) ↑ε'
    -/
    dsimp [norm]
    /-
      case intro.intro.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      f : CauSeq (Padic p) Norm.norm
      cau_seq_norm_e : IsCauSeq ⇑padicNormE ↑f
      q : Padic p
      hq : ∀ (ε : Rat), GT.gt ε 0 → Exists fun N => ∀ (i : Nat), GE.ge i N → LT.lt ( …
      ε : Real
      hε : GT.gt ε 0
      ε' : Rat
      hε' : And (LT.lt 0 ε') (LT.lt (↑ε') ε)
      N : Nat
      hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (↑⟨↑f, cau_seq_norm …
      i : Nat
      hi : GE.ge i N
      h : LT.lt (padicNormE (HSub.hSub (↑⟨↑f, cau_seq_norm_e⟩ i) q)) ε'
      ⊢ LT.lt ↑(padicNormE (HSub.hSub (↑f i) q)) ↑ε'
    -/
    exact mod_cast h
    /-
      🎉 no goals
    -/


theorem padicNormE_lim_le {f : CauSeq ℚ_[p] norm} {a : ℝ} (ha : 0 < a) (hf : ∀ i, ‖f i‖ ≤ a) :
    ‖f.lim‖ ≤ a := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    f : CauSeq (Padic p) Norm.norm
    a : Real
    ha : LT.lt 0 a
    hf : ∀ (i : Nat), LE.le (Norm.norm (↑f i)) a
    ⊢ LE.le (Norm.norm f.lim) a
  -/
  obtain ⟨N, hN⟩ := Setoid.symm (CauSeq.equiv_lim f) _ ha
  calc
    ‖f.lim‖ = ‖f.lim - f N + f N‖ := by simp
    _ ≤ max ‖f.lim - f N‖ ‖f N‖ := padicNormE.nonarchimedean _ _
    _ ≤ a := max_le (le_of_lt (hN _ le_rfl)) (hf _)


instance : CompleteSpace ℚ_[p] := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ CompleteSpace (Padic p)
  -/
  apply complete_of_cauchySeq_tendsto
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ ∀ (u : Nat → Padic p), CauchySeq u → Exists fun a => Filter.Tendsto u Filter …
  -/
  intro u hu
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    u : Nat → Padic p
    hu : CauchySeq u
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  let c : CauSeq ℚ_[p] norm := ⟨u, Metric.cauchySeq_iff'.mp hu⟩
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    u : Nat → Padic p
    hu : CauchySeq u
    c : CauSeq (Padic p) Norm.norm := ⟨u, ⋯⟩
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  refine ⟨c.lim, fun s h ↦ ?_⟩
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    u : Nat → Padic p
    hu : CauchySeq u
    c : CauSeq (Padic p) Norm.norm := ⟨u, ⋯⟩
    s : Set (Padic p)
    h : Membership.mem (nhds c.lim) s
    ⊢ Membership.mem (Filter.map u Filter.atTop) s
  -/
  rcases Metric.mem_nhds_iff.1 h with ⟨ε, ε0, hε⟩
  /-
    case a.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    u : Nat → Padic p
    hu : CauchySeq u
    c : CauSeq (Padic p) Norm.norm := ⟨u, ⋯⟩
    s : Set (Padic p)
    h : Membership.mem (nhds c.lim) s
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball c.lim ε) s
    ⊢ Membership.mem (Filter.map u Filter.atTop) s
  -/
  have := c.equiv_lim ε ε0
  /-
    case a.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    u : Nat → Padic p
    hu : CauchySeq u
    c : CauSeq (Padic p) Norm.norm := ⟨u, ⋯⟩
    s : Set (Padic p)
    h : Membership.mem (nhds c.lim) s
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball c.lim ε) s
    this : Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (↑(HSub.hSub  …
    ⊢ Membership.mem (Filter.map u Filter.atTop) s
  -/
  simp only [mem_map, mem_atTop_sets, mem_setOf_eq]
  /-
    case a.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    u : Nat → Padic p
    hu : CauchySeq u
    c : CauSeq (Padic p) Norm.norm := ⟨u, ⋯⟩
    s : Set (Padic p)
    h : Membership.mem (nhds c.lim) s
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball c.lim ε) s
    this : Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (↑(HSub.hSub  …
    ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem (Set.preimage u s) b
  -/
  exact this.imp fun N hN n hn ↦ hε (hN n hn)
  /-
    🎉 no goals
  -/


/-- `Padic.valuation` lifts the `p`-adic valuation on rationals to `ℚ_[p]`. -/
def valuation : ℚ_[p] → ℤ :=
  Quotient.lift (@PadicSeq.valuation p _) fun f g h ↦ by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      f g : CauSeq Rat (padicNorm p)
      h : HasEquiv.Equiv f g
      ⊢ Eq (PadicSeq.valuation f) (PadicSeq.valuation g)
    -/
    by_cases hf : f ≈ 0
      /-
        case pos
        p : Nat
        hp : Fact (Nat.Prime p)
        f g : CauSeq Rat (padicNorm p)
        h : HasEquiv.Equiv f g
        hf : HasEquiv.Equiv f 0
        ⊢ Eq (PadicSeq.valuation f) (PadicSeq.valuation g)
      -/
    · have hg : g ≈ 0 := Setoid.trans (Setoid.symm h) hf
      /-
        case pos
        p : Nat
        hp : Fact (Nat.Prime p)
        f g : CauSeq Rat (padicNorm p)
        h : HasEquiv.Equiv f g
        hf : HasEquiv.Equiv f 0
        hg : HasEquiv.Equiv g 0
        ⊢ Eq (PadicSeq.valuation f) (PadicSeq.valuation g)
      -/
      simp [hf, hg, PadicSeq.valuation]
      /-
        🎉 no goals
      -/
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        f g : CauSeq Rat (padicNorm p)
        h : HasEquiv.Equiv f g
        hf : Not (HasEquiv.Equiv f 0)
        ⊢ Eq (PadicSeq.valuation f) (PadicSeq.valuation g)
      -/
    · have hg : ¬g ≈ 0 := fun hg ↦ hf (Setoid.trans h hg)
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        f g : CauSeq Rat (padicNorm p)
        h : HasEquiv.Equiv f g
        hf : Not (HasEquiv.Equiv f 0)
        hg : Not (HasEquiv.Equiv g 0)
        ⊢ Eq (PadicSeq.valuation f) (PadicSeq.valuation g)
      -/
      rw [PadicSeq.val_eq_iff_norm_eq hf hg]
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        f g : CauSeq Rat (padicNorm p)
        h : HasEquiv.Equiv f g
        hf : Not (HasEquiv.Equiv f 0)
        hg : Not (HasEquiv.Equiv g 0)
        ⊢ Eq (PadicSeq.norm f) (PadicSeq.norm g)
      -/
      exact PadicSeq.norm_equiv h
      /-
        🎉 no goals
      -/


@[simp]
theorem valuation_zero : valuation (0 : ℚ_[p]) = 0 :=
  dif_pos ((const_equiv p).2 rfl)


theorem norm_eq_zpow_neg_valuation {x : ℚ_[p]} : x ≠ 0 → ‖x‖ = (p : ℝ) ^ (-x.valuation) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    ⊢ Ne x 0 → Eq (Norm.norm x) (HPow.hPow (↑p) (Neg.neg x.valuation))
  -/
  refine Quotient.inductionOn' x fun f hf => ?_
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    f : CauSeq Rat (padicNorm p)
    hf : Ne (Quotient.mk'' f) 0
    ⊢ Eq (Norm.norm (Quotient.mk'' f)) (HPow.hPow (↑p) (Neg.neg (Padic.valuation ( …
  -/
  change (PadicSeq.norm _ : ℝ) = (p : ℝ) ^ (-PadicSeq.valuation _)
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    f : CauSeq Rat (padicNorm p)
    hf : Ne (Quotient.mk'' f) 0
    ⊢ Eq (↑(PadicSeq.norm f)) (HPow.hPow (↑p) (Neg.neg (PadicSeq.valuation f)))
  -/
  rw [PadicSeq.norm_eq_zpow_neg_valuation]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Padic p
      f : CauSeq Rat (padicNorm p)
      hf : Ne (Quotient.mk'' f) 0
      ⊢ Eq (↑(HPow.hPow (↑p) (Neg.neg (PadicSeq.valuation f)))) (HPow.hPow (↑p) (Neg …
    -/
  · rw [Rat.cast_zpow, Rat.cast_natCast]
    /-
      🎉 no goals
    -/
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Padic p
      f : CauSeq Rat (padicNorm p)
      hf : Ne (Quotient.mk'' f) 0
      ⊢ Not (HasEquiv.Equiv f 0)
    -/
  · apply CauSeq.not_limZero_of_not_congr_zero
    -- Porting note: was `contrapose! hf`
    /-
      case hf
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Padic p
      f : CauSeq Rat (padicNorm p)
      hf : Ne (Quotient.mk'' f) 0
      ⊢ Not (HasEquiv.Equiv (HSub.hSub f 0) 0)
    -/
    intro hf'
    /-
      case hf
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Padic p
      f : CauSeq Rat (padicNorm p)
      hf : Ne (Quotient.mk'' f) 0
      hf' : HasEquiv.Equiv (HSub.hSub f 0) 0
      ⊢ False
    -/
    apply hf
    /-
      case hf
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Padic p
      f : CauSeq Rat (padicNorm p)
      hf : Ne (Quotient.mk'' f) 0
      hf' : HasEquiv.Equiv (HSub.hSub f 0) 0
      ⊢ Eq (Quotient.mk'' f) 0
    -/
    apply Quotient.sound
    /-
      case hf.a
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Padic p
      f : CauSeq Rat (padicNorm p)
      hf : Ne (Quotient.mk'' f) 0
      hf' : HasEquiv.Equiv (HSub.hSub f 0) 0
      ⊢ HasEquiv.Equiv f (CauSeq.const (padicNorm p) 0)
    -/
    simpa using hf'
    /-
      🎉 no goals
    -/


@[simp]
lemma valuation_ratCast (q : ℚ) : valuation (q : ℚ_[p]) = padicValRat p q := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    ⊢ Eq (↑q).valuation (padicValRat p q)
  -/
  rcases eq_or_ne q 0 with rfl | hq
    /-
      case inl
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ Eq (↑0).valuation (padicValRat p 0)
    -/
  · simp only [Rat.cast_zero, valuation_zero, padicValRat.zero]
    /-
      🎉 no goals
    -/
  refine neg_injective ((zpow_right_strictMono₀ (mod_cast hp.out.one_lt)).injective
    <| (norm_eq_zpow_neg_valuation (mod_cast hq)).symm.trans ?_)
  /-
    case inr
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    hq : Ne q 0
    ⊢ Eq (Norm.norm ↑q) (HPow.hPow (↑p) (Neg.neg (padicValRat p q)))
  -/
  rw [padicNormE.eq_padicNorm, ← Rat.cast_natCast, ← Rat.cast_zpow, Rat.cast_inj]
  /-
    case inr
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    hq : Ne q 0
    ⊢ Eq (padicNorm p q) (HPow.hPow (↑p) (Neg.neg (padicValRat p q)))
  -/
  exact padicNorm.eq_zpow_of_nonzero hq
  /-
    🎉 no goals
  -/


@[simp]
lemma valuation_intCast (n : ℤ) : valuation (n : ℚ_[p]) = padicValInt p n := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Int
    ⊢ Eq (↑n).valuation ↑(padicValInt p n)
  -/
  rw [← Rat.cast_intCast, valuation_ratCast, padicValRat.of_int]
  /-
    🎉 no goals
  -/


@[simp]
lemma valuation_natCast (n : ℕ) : valuation (n : ℚ_[p]) = padicValNat p n := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (↑n).valuation ↑(padicValNat p n)
  -/
  rw [← Rat.cast_natCast, valuation_ratCast, padicValRat.of_nat]
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
lemma valuation_ofNat (n : ℕ) [n.AtLeastTwo] :
    valuation (no_index (OfNat.ofNat n : ℚ_[p])) = padicValNat p n :=
  valuation_natCast n


@[simp]
lemma valuation_one : valuation (1 : ℚ_[p]) = 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (Padic.valuation 1) 0
  -/
  rw [← Nat.cast_one, valuation_natCast, padicValNat.one, cast_zero]
  /-
    🎉 no goals
  -/

-- not @[simp], since simp can prove it

lemma valuation_p : valuation (p : ℚ_[p]) = 1 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (↑p).valuation 1
  -/
  rw [valuation_natCast, padicValNat_self, cast_one]
  /-
    🎉 no goals
  -/


theorem le_valuation_add {x y : ℚ_[p]} (hxy : x + y ≠ 0) :
    min x.valuation y.valuation ≤ (x + y).valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : Padic p
    hxy : Ne (HAdd.hAdd x y) 0
    ⊢ LE.le (Min.min x.valuation y.valuation) (HAdd.hAdd x y).valuation
  -/
  by_cases hx : x = 0
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      x y : Padic p
      hxy : Ne (HAdd.hAdd x y) 0
      hx : Eq x 0
      ⊢ LE.le (Min.min x.valuation y.valuation) (HAdd.hAdd x y).valuation
    -/
  · simpa only [hx, zero_add] using min_le_right _ _
    /-
      🎉 no goals
    -/
  /-
    case neg
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : Padic p
    hxy : Ne (HAdd.hAdd x y) 0
    hx : Not (Eq x 0)
    ⊢ LE.le (Min.min x.valuation y.valuation) (HAdd.hAdd x y).valuation
  -/
  by_cases hy : y = 0
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      x y : Padic p
      hxy : Ne (HAdd.hAdd x y) 0
      hx : Not (Eq x 0)
      hy : Eq y 0
      ⊢ LE.le (Min.min x.valuation y.valuation) (HAdd.hAdd x y).valuation
    -/
  · simpa only [hy, add_zero] using min_le_left _ _
    /-
      🎉 no goals
    -/
  /-
    case neg
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : Padic p
    hxy : Ne (HAdd.hAdd x y) 0
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ LE.le (Min.min x.valuation y.valuation) (HAdd.hAdd x y).valuation
  -/
  have : ‖x + y‖ ≤ max ‖x‖ ‖y‖ := padicNormE.nonarchimedean x y
  simpa only [norm_eq_zpow_neg_valuation hxy, norm_eq_zpow_neg_valuation hx,
    norm_eq_zpow_neg_valuation hy, le_max_iff,
    zpow_le_zpow_iff_right₀ (mod_cast hp.out.one_lt : 1 < (p : ℝ)), neg_le_neg_iff, ← min_le_iff]


@[simp]
lemma valuation_mul {x y : ℚ_[p]} (hx : x ≠ 0) (hy : y ≠ 0) :
    (x * y).valuation = x.valuation + y.valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : Padic p
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (HMul.hMul x y).valuation (HAdd.hAdd x.valuation y.valuation)
  -/
  have h_norm : ‖x * y‖ = ‖x‖ * ‖y‖ := norm_mul x y
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : Padic p
    hx : Ne x 0
    hy : Ne y 0
    h_norm : Eq (Norm.norm (HMul.hMul x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
    ⊢ Eq (HMul.hMul x y).valuation (HAdd.hAdd x.valuation y.valuation)
  -/
  have hp_ne_one : (p : ℝ) ≠ 1 := mod_cast (Fact.out : p.Prime).ne_one
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : Padic p
    hx : Ne x 0
    hy : Ne y 0
    h_norm : Eq (Norm.norm (HMul.hMul x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
    hp_ne_one : Ne (↑p) 1
    ⊢ Eq (HMul.hMul x y).valuation (HAdd.hAdd x.valuation y.valuation)
  -/
  have hp_pos : (0 : ℝ) < p := mod_cast NeZero.pos _
  rwa [norm_eq_zpow_neg_valuation hx, norm_eq_zpow_neg_valuation hy,
    norm_eq_zpow_neg_valuation (mul_ne_zero hx hy), ← zpow_add₀ hp_pos.ne',
    zpow_right_inj₀ hp_pos hp_ne_one, ← neg_add, neg_inj] at h_norm


@[simp]
lemma valuation_inv (x : ℚ_[p]) : x⁻¹.valuation = -x.valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    ⊢ Eq (Inv.inv x).valuation (Neg.neg x.valuation)
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inl
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ Eq (Inv.inv 0).valuation (Neg.neg (Padic.valuation 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    hx : Ne x 0
    ⊢ Eq (Inv.inv x).valuation (Neg.neg x.valuation)
  -/
  have h_norm : ‖x⁻¹‖ = ‖x‖⁻¹ := norm_inv x
  /-
    case inr
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    hx : Ne x 0
    h_norm : Eq (Norm.norm (Inv.inv x)) (Inv.inv (Norm.norm x))
    ⊢ Eq (Inv.inv x).valuation (Neg.neg x.valuation)
  -/
  have hp_ne_one : (p : ℝ) ≠ 1 := mod_cast (Fact.out : p.Prime).ne_one
  /-
    case inr
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    hx : Ne x 0
    h_norm : Eq (Norm.norm (Inv.inv x)) (Inv.inv (Norm.norm x))
    hp_ne_one : Ne (↑p) 1
    ⊢ Eq (Inv.inv x).valuation (Neg.neg x.valuation)
  -/
  have hp_pos : (0 : ℝ) < p := mod_cast NeZero.pos _
  rwa [norm_eq_zpow_neg_valuation hx, norm_eq_zpow_neg_valuation <| inv_ne_zero hx,
    ← zpow_neg, zpow_right_inj₀ hp_pos hp_ne_one, neg_inj] at h_norm


@[simp]
lemma valuation_pow (x : ℚ_[p]) : ∀ n : ℕ, (x ^ n).valuation = n * x.valuation
            /-
              p : Nat
              hp : Fact (Nat.Prime p)
              x : Padic p
              ⊢ Eq (HPow.hPow x 0).valuation (HMul.hMul (↑0) x.valuation)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Padic p
      n : Nat
      ⊢ Eq (HPow.hPow x (HAdd.hAdd n 1)).valuation (HMul.hMul (↑(HAdd.hAdd n 1)) x.v …
    -/
    obtain rfl | hx := eq_or_ne x 0
      /-
        case inl
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Nat
        ⊢ Eq (HPow.hPow 0 (HAdd.hAdd n 1)).valuation (HMul.hMul (↑(HAdd.hAdd n 1)) (Pa …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case inr
        p : Nat
        hp : Fact (Nat.Prime p)
        x : Padic p
        n : Nat
        hx : Ne x 0
        ⊢ Eq (HPow.hPow x (HAdd.hAdd n 1)).valuation (HMul.hMul (↑(HAdd.hAdd n 1)) x.v …
      -/
    · simp [pow_succ, hx, valuation_mul, valuation_pow, _root_.add_one_mul]
      /-
        🎉 no goals
      -/


@[simp]
lemma valuation_zpow (x : ℚ_[p]) : ∀ n : ℤ, (x ^ n).valuation = n * x.valuation
                  /-
                    p : Nat
                    hp : Fact (Nat.Prime p)
                    x : Padic p
                    n : Nat
                    ⊢ Eq (HPow.hPow x ↑n).valuation (HMul.hMul (↑n) x.valuation)
                  -/
  | (n : ℕ) => by simp
                  /-
                    🎉 no goals
                  -/
                     /-
                       p : Nat
                       hp : Fact (Nat.Prime p)
                       x : Padic p
                       n : Nat
                       ⊢ Eq (HPow.hPow x (Int.negSucc n)).valuation (HMul.hMul (Int.negSucc n) x.valu …
                     -/
  | .negSucc n => by simp [← neg_mul]; simp [Int.negSucc_eq]
                                       /-
                                         🎉 no goals
                                       -/


@[deprecated (since := "2024-12-10")] alias valuation_map_add := le_valuation_add

@[deprecated (since := "2024-12-10")] alias valuation_map_mul := valuation_mul


open Classical in
/-- The additive `p`-adic valuation on `ℚ_[p]`, with values in `WithTop ℤ`. -/
def addValuationDef : ℚ_[p] → WithTop ℤ :=
  fun x ↦ if x = 0 then ⊤ else x.valuation


@[simp]
theorem AddValuation.map_zero : addValuationDef (0 : ℚ_[p]) = ⊤ := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (Padic.addValuationDef 0) Top.top
  -/
  rw [addValuationDef, if_pos rfl]
  /-
    🎉 no goals
  -/


@[simp]
theorem AddValuation.map_one : addValuationDef (1 : ℚ_[p]) = 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (Padic.addValuationDef 1) 0
  -/
  rw [addValuationDef, if_neg one_ne_zero, valuation_one, WithTop.coe_zero]
  /-
    🎉 no goals
  -/


theorem AddValuation.map_mul (x y : ℚ_[p]) :
    addValuationDef (x * y : ℚ_[p]) = addValuationDef x + addValuationDef y := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : Padic p
    ⊢ Eq (HMul.hMul x y).addValuationDef (HAdd.hAdd x.addValuationDef y.addValuati …
  -/
  simp only [addValuationDef]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : Padic p
    ⊢ Eq (ite (Eq (HMul.hMul x y) 0) Top.top ↑(HMul.hMul x y).valuation) (HAdd.hAd …
  -/
  by_cases hx : x = 0
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      x y : Padic p
      hx : Eq x 0
      ⊢ Eq (ite (Eq (HMul.hMul x y) 0) Top.top ↑(HMul.hMul x y).valuation) (HAdd.hAd …
    -/
  · rw [hx, if_pos rfl, zero_mul, if_pos rfl, WithTop.top_add]
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : Nat
      hp : Fact (Nat.Prime p)
      x y : Padic p
      hx : Not (Eq x 0)
      ⊢ Eq (ite (Eq (HMul.hMul x y) 0) Top.top ↑(HMul.hMul x y).valuation) (HAdd.hAd …
    -/
  · by_cases hy : y = 0
      /-
        case pos
        p : Nat
        hp : Fact (Nat.Prime p)
        x y : Padic p
        hx : Not (Eq x 0)
        hy : Eq y 0
        ⊢ Eq (ite (Eq (HMul.hMul x y) 0) Top.top ↑(HMul.hMul x y).valuation) (HAdd.hAd …
      -/
    · rw [hy, if_pos rfl, mul_zero, if_pos rfl, WithTop.add_top]
      /-
        🎉 no goals
      -/
    · rw [if_neg hx, if_neg hy, if_neg (mul_ne_zero hx hy), ← WithTop.coe_add, WithTop.coe_eq_coe,
        valuation_mul hx hy]


theorem AddValuation.map_add (x y : ℚ_[p]) :
    min (addValuationDef x) (addValuationDef y) ≤ addValuationDef (x + y : ℚ_[p]) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : Padic p
    ⊢ LE.le (Min.min x.addValuationDef y.addValuationDef) (HAdd.hAdd x y).addValua …
  -/
  simp only [addValuationDef]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : Padic p
    ⊢ LE.le (Min.min (ite (Eq x 0) Top.top ↑x.valuation) (ite (Eq y 0) Top.top ↑y. …
  -/
  by_cases hxy : x + y = 0
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      x y : Padic p
      hxy : Eq (HAdd.hAdd x y) 0
      ⊢ LE.le (Min.min (ite (Eq x 0) Top.top ↑x.valuation) (ite (Eq y 0) Top.top ↑y. …
    -/
  · rw [hxy, if_pos rfl]
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      x y : Padic p
      hxy : Eq (HAdd.hAdd x y) 0
      ⊢ LE.le (Min.min (ite (Eq x 0) Top.top ↑x.valuation) (ite (Eq y 0) Top.top ↑y. …
    -/
    exact le_top
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : Nat
      hp : Fact (Nat.Prime p)
      x y : Padic p
      hxy : Not (Eq (HAdd.hAdd x y) 0)
      ⊢ LE.le (Min.min (ite (Eq x 0) Top.top ↑x.valuation) (ite (Eq y 0) Top.top ↑y. …
    -/
  · by_cases hx : x = 0
      /-
        case pos
        p : Nat
        hp : Fact (Nat.Prime p)
        x y : Padic p
        hxy : Not (Eq (HAdd.hAdd x y) 0)
        hx : Eq x 0
        ⊢ LE.le (Min.min (ite (Eq x 0) Top.top ↑x.valuation) (ite (Eq y 0) Top.top ↑y. …
      -/
    · rw [hx, if_pos rfl, min_eq_right, zero_add]
      /-
        case pos
        p : Nat
        hp : Fact (Nat.Prime p)
        x y : Padic p
        hxy : Not (Eq (HAdd.hAdd x y) 0)
        hx : Eq x 0
        ⊢ LE.le (ite (Eq y 0) Top.top ↑y.valuation) Top.top
      -/
      exact le_top
      /-
        🎉 no goals
      -/
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        x y : Padic p
        hxy : Not (Eq (HAdd.hAdd x y) 0)
        hx : Not (Eq x 0)
        ⊢ LE.le (Min.min (ite (Eq x 0) Top.top ↑x.valuation) (ite (Eq y 0) Top.top ↑y. …
      -/
    · by_cases hy : y = 0
        /-
          case pos
          p : Nat
          hp : Fact (Nat.Prime p)
          x y : Padic p
          hxy : Not (Eq (HAdd.hAdd x y) 0)
          hx : Not (Eq x 0)
          hy : Eq y 0
          ⊢ LE.le (Min.min (ite (Eq x 0) Top.top ↑x.valuation) (ite (Eq y 0) Top.top ↑y. …
        -/
      · rw [hy, if_pos rfl, min_eq_left, add_zero]
        /-
          case pos
          p : Nat
          hp : Fact (Nat.Prime p)
          x y : Padic p
          hxy : Not (Eq (HAdd.hAdd x y) 0)
          hx : Not (Eq x 0)
          hy : Eq y 0
          ⊢ LE.le (ite (Eq x 0) Top.top ↑x.valuation) Top.top
        -/
        exact le_top
        /-
          🎉 no goals
        -/
        /-
          case neg
          p : Nat
          hp : Fact (Nat.Prime p)
          x y : Padic p
          hxy : Not (Eq (HAdd.hAdd x y) 0)
          hx : Not (Eq x 0)
          hy : Not (Eq y 0)
          ⊢ LE.le (Min.min (ite (Eq x 0) Top.top ↑x.valuation) (ite (Eq y 0) Top.top ↑y. …
        -/
      · rw [if_neg hx, if_neg hy, if_neg hxy, ← WithTop.coe_min, WithTop.coe_le_coe]
        /-
          case neg
          p : Nat
          hp : Fact (Nat.Prime p)
          x y : Padic p
          hxy : Not (Eq (HAdd.hAdd x y) 0)
          hx : Not (Eq x 0)
          hy : Not (Eq y 0)
          ⊢ LE.le (Min.min x.valuation y.valuation) (HAdd.hAdd x y).valuation
        -/
        exact le_valuation_add hxy
        /-
          🎉 no goals
        -/


/-- The additive `p`-adic valuation on `ℚ_[p]`, as an `addValuation`. -/
def addValuation : AddValuation ℚ_[p] (WithTop ℤ) :=
  AddValuation.of addValuationDef AddValuation.map_zero AddValuation.map_one AddValuation.map_add
    AddValuation.map_mul


@[simp]
theorem addValuation.apply {x : ℚ_[p]} (hx : x ≠ 0) :
    Padic.addValuation x = (x.valuation : WithTop ℤ) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    hx : Ne x 0
    ⊢ Eq (Padic.addValuation x) ↑x.valuation
  -/
  simp only [Padic.addValuation, AddValuation.of_apply, addValuationDef, if_neg hx]
  /-
    🎉 no goals
  -/


theorem norm_le_pow_iff_norm_lt_pow_add_one (x : ℚ_[p]) (n : ℤ) :
    ‖x‖ ≤ (p : ℝ) ^ n ↔ ‖x‖ < (p : ℝ) ^ (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    n : Int
    ⊢ Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) n)) (LT.lt (Norm.norm x) (HPow.hPow …
  -/
  have aux (n : ℤ) : 0 < ((p : ℝ) ^ n) := zpow_pos (mod_cast hp.1.pos) _
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    n : Int
    aux : ∀ (n : Int), LT.lt 0 (HPow.hPow (↑p) n)
    ⊢ Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) n)) (LT.lt (Norm.norm x) (HPow.hPow …
  -/
  by_cases hx0 : x = 0
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Padic p
      n : Int
      aux : ∀ (n : Int), LT.lt 0 (HPow.hPow (↑p) n)
      hx0 : Eq x 0
      ⊢ Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) n)) (LT.lt (Norm.norm x) (HPow.hPow …
    -/
  · simp [hx0, norm_zero, aux, le_of_lt (aux _)]
    /-
      🎉 no goals
    -/
  /-
    case neg
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    n : Int
    aux : ∀ (n : Int), LT.lt 0 (HPow.hPow (↑p) n)
    hx0 : Not (Eq x 0)
    ⊢ Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) n)) (LT.lt (Norm.norm x) (HPow.hPow …
  -/
  rw [norm_eq_zpow_neg_valuation hx0]
  /-
    case neg
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    n : Int
    aux : ∀ (n : Int), LT.lt 0 (HPow.hPow (↑p) n)
    hx0 : Not (Eq x 0)
    ⊢ Iff (LE.le (HPow.hPow (↑p) (Neg.neg x.valuation)) (HPow.hPow (↑p) n)) (LT.lt …
  -/
  have h1p : 1 < (p : ℝ) := mod_cast hp.1.one_lt
  /-
    case neg
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    n : Int
    aux : ∀ (n : Int), LT.lt 0 (HPow.hPow (↑p) n)
    hx0 : Not (Eq x 0)
    h1p : LT.lt 1 ↑p
    ⊢ Iff (LE.le (HPow.hPow (↑p) (Neg.neg x.valuation)) (HPow.hPow (↑p) n)) (LT.lt …
  -/
  have H := zpow_right_strictMono₀ h1p
  /-
    case neg
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    n : Int
    aux : ∀ (n : Int), LT.lt 0 (HPow.hPow (↑p) n)
    hx0 : Not (Eq x 0)
    h1p : LT.lt 1 ↑p
    H : StrictMono fun n => HPow.hPow (↑p) n
    ⊢ Iff (LE.le (HPow.hPow (↑p) (Neg.neg x.valuation)) (HPow.hPow (↑p) n)) (LT.lt …
  -/
  rw [H.le_iff_le, H.lt_iff_lt, Int.lt_add_one_iff]
  /-
    🎉 no goals
  -/


theorem norm_lt_pow_iff_norm_le_pow_sub_one (x : ℚ_[p]) (n : ℤ) :
    ‖x‖ < (p : ℝ) ^ n ↔ ‖x‖ ≤ (p : ℝ) ^ (n - 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    n : Int
    ⊢ Iff (LT.lt (Norm.norm x) (HPow.hPow (↑p) n)) (LE.le (Norm.norm x) (HPow.hPow …
  -/
  rw [norm_le_pow_iff_norm_lt_pow_add_one, sub_add_cancel]
  /-
    🎉 no goals
  -/


theorem norm_le_one_iff_val_nonneg (x : ℚ_[p]) : ‖x‖ ≤ 1 ↔ 0 ≤ x.valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : Padic p
    ⊢ Iff (LE.le (Norm.norm x) 1) (LE.le 0 x.valuation)
  -/
  by_cases hx : x = 0
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Padic p
      hx : Eq x 0
      ⊢ Iff (LE.le (Norm.norm x) 1) (LE.le 0 x.valuation)
    -/
  · simp only [hx, norm_zero, valuation_zero, zero_le_one, le_refl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Padic p
      hx : Not (Eq x 0)
      ⊢ Iff (LE.le (Norm.norm x) 1) (LE.le 0 x.valuation)
    -/
  · rw [norm_eq_zpow_neg_valuation hx, ← zpow_zero (p : ℝ), zpow_le_zpow_iff_right₀, neg_nonpos]
    /-
      case neg
      p : Nat
      hp : Fact (Nat.Prime p)
      x : Padic p
      hx : Not (Eq x 0)
      ⊢ LT.lt 1 ↑p
    -/
    exact Nat.one_lt_cast.2 (Nat.Prime.one_lt' p).1
    /-
      🎉 no goals
    -/


