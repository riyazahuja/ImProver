local prefix:100 "s" => cs.simple

local prefix:100 "π" => cs.wordProd

local prefix:100 "ℓ" => cs.length


/-- `t : W` is a *reflection* of the Coxeter system `cs` if it is of the form
$w s_i w^{-1}$, where $w \in W$ and $s_i$ is a simple reflection. -/
def IsReflection (t : W) : Prop := ∃ w i, t = w * s i * w⁻¹


                                                                  /-
                                                                    B : Type u_1
                                                                    W : Type u_2
                                                                    inst✝ : Group W
                                                                    M : CoxeterMatrix B
                                                                    cs : CoxeterSystem M W
                                                                    i : B
                                                                    ⊢ cs.IsReflection (cs.simple i)
                                                                  -/
theorem isReflection_simple (i : B) : cs.IsReflection (s i) := by use 1, i; simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem pow_two : t ^ 2 = 1 := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    t : W
    ht : cs.IsReflection t
    ⊢ Eq (HPow.hPow t 2) 1
  -/
  rcases ht with ⟨w, i, rfl⟩
  /-
    case intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Eq (HPow.hPow (HMul.hMul (HMul.hMul w (cs.simple i)) (Inv.inv w)) 2) 1
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mul_self : t * t = 1 := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    t : W
    ht : cs.IsReflection t
    ⊢ Eq (HMul.hMul t t) 1
  -/
  rcases ht with ⟨w, i, rfl⟩
  /-
    case intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul w (cs.simple i)) (Inv.inv w)) (HMul.hMul …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem inv : t⁻¹ = t := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    t : W
    ht : cs.IsReflection t
    ⊢ Eq (Inv.inv t) t
  -/
  rcases ht with ⟨w, i, rfl⟩
  /-
    case intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Eq (Inv.inv (HMul.hMul (HMul.hMul w (cs.simple i)) (Inv.inv w))) (HMul.hMul  …
  -/
  simp [mul_assoc]
  /-
    🎉 no goals
  -/


                                                     /-
                                                       B : Type u_1
                                                       W : Type u_2
                                                       inst✝ : Group W
                                                       M : CoxeterMatrix B
                                                       cs : CoxeterSystem M W
                                                       t : W
                                                       ht : cs.IsReflection t
                                                       ⊢ cs.IsReflection (Inv.inv t)
                                                     -/
theorem isReflection_inv : cs.IsReflection t⁻¹ := by rwa [ht.inv]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem odd_length : Odd (ℓ t) := by
  suffices cs.lengthParity t = Multiplicative.ofAdd 1 by
    simpa [lengthParity_eq_ofAdd_length, ZMod.eq_one_iff_odd]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    t : W
    ht : cs.IsReflection t
    ⊢ Eq (cs.lengthParity t) (Multiplicative.ofAdd 1)
  -/
  rcases ht with ⟨w, i, rfl⟩
  /-
    case intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Eq (cs.lengthParity (HMul.hMul (HMul.hMul w (cs.simple i)) (Inv.inv w))) (Mu …
  -/
  simp [lengthParity_simple]
  /-
    🎉 no goals
  -/


theorem length_mul_left_ne (w : W) : ℓ (w * t) ≠ ℓ w := by
  suffices cs.lengthParity (w * t) ≠ cs.lengthParity w by
    contrapose! this
    simp only [lengthParity_eq_ofAdd_length, this]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    t : W
    ht : cs.IsReflection t
    w : W
    ⊢ Ne (cs.lengthParity (HMul.hMul w t)) (cs.lengthParity w)
  -/
  rcases ht with ⟨w, i, rfl⟩
  /-
    case intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w✝ w : W
    i : B
    ⊢ Ne (cs.lengthParity (HMul.hMul w✝ (HMul.hMul (HMul.hMul w (cs.simple i)) (In …
  -/
  simp [lengthParity_simple]
  /-
    🎉 no goals
  -/


theorem length_mul_right_ne (w : W) : ℓ (t * w) ≠ ℓ w := by
  suffices cs.lengthParity (t * w) ≠ cs.lengthParity w by
    contrapose! this
    simp only [lengthParity_eq_ofAdd_length, this]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    t : W
    ht : cs.IsReflection t
    w : W
    ⊢ Ne (cs.lengthParity (HMul.hMul t w)) (cs.lengthParity w)
  -/
  rcases ht with ⟨w, i, rfl⟩
  /-
    case intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w✝ w : W
    i : B
    ⊢ Ne (cs.lengthParity (HMul.hMul (HMul.hMul (HMul.hMul w (cs.simple i)) (Inv.i …
  -/
  simp [lengthParity_simple]
  /-
    🎉 no goals
  -/


theorem conj (w : W) : cs.IsReflection (w * t * w⁻¹) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    t : W
    ht : cs.IsReflection t
    w : W
    ⊢ cs.IsReflection (HMul.hMul (HMul.hMul w t) (Inv.inv w))
  -/
  obtain ⟨u, i, rfl⟩ := ht
  /-
    case intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w u : W
    i : B
    ⊢ cs.IsReflection (HMul.hMul (HMul.hMul w (HMul.hMul (HMul.hMul u (cs.simple i …
  -/
  use w * u, i
  /-
    case h
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w u : W
    i : B
    ⊢ Eq (HMul.hMul (HMul.hMul w (HMul.hMul (HMul.hMul u (cs.simple i)) (Inv.inv u …
  -/
  group
  /-
    🎉 no goals
  -/


@[simp]
theorem isReflection_conj_iff (w t : W) :
    cs.IsReflection (w * t * w⁻¹) ↔ cs.IsReflection t := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w t : W
    ⊢ Iff (cs.IsReflection (HMul.hMul (HMul.hMul w t) (Inv.inv w))) (cs.IsReflecti …
  -/
  constructor
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w t : W
      ⊢ cs.IsReflection (HMul.hMul (HMul.hMul w t) (Inv.inv w)) → cs.IsReflection t
    -/
  · intro h
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w t : W
      h : cs.IsReflection (HMul.hMul (HMul.hMul w t) (Inv.inv w))
      ⊢ cs.IsReflection t
    -/
    simpa [← mul_assoc] using h.conj w⁻¹
    /-
      🎉 no goals
    -/
    /-
      case mpr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w t : W
      ⊢ cs.IsReflection t → cs.IsReflection (HMul.hMul (HMul.hMul w t) (Inv.inv w))
    -/
  · exact IsReflection.conj (w := w)
    /-
      🎉 no goals
    -/


/-- The proposition that `t` is a right inversion of `w`; i.e., `t` is a reflection and
$\ell (w t) < \ell(w)$. -/
def IsRightInversion (w t : W) : Prop := cs.IsReflection t ∧ ℓ (w * t) < ℓ w


/-- The proposition that `t` is a left inversion of `w`; i.e., `t` is a reflection and
$\ell (t w) < \ell(w)$. -/
def IsLeftInversion (w t : W) : Prop := cs.IsReflection t ∧ ℓ (t * w) < ℓ w


theorem isRightInversion_inv_iff {w t : W} :
    cs.IsRightInversion w⁻¹ t ↔ cs.IsLeftInversion w t := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w t : W
    ⊢ Iff (cs.IsRightInversion (Inv.inv w) t) (cs.IsLeftInversion w t)
  -/
  apply and_congr_right
  /-
    case h
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w t : W
    ⊢ cs.IsReflection t → Iff (LT.lt (cs.length (HMul.hMul (Inv.inv w) t)) (cs.len …
  -/
  intro ht
  /-
    case h
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w t : W
    ht : cs.IsReflection t
    ⊢ Iff (LT.lt (cs.length (HMul.hMul (Inv.inv w) t)) (cs.length (Inv.inv w))) (L …
  -/
  rw [← length_inv, mul_inv_rev, inv_inv, ht.inv, cs.length_inv w]
  /-
    🎉 no goals
  -/


theorem isLeftInversion_inv_iff {w t : W} :
    cs.IsLeftInversion w⁻¹ t ↔ cs.IsRightInversion w t := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w t : W
    ⊢ Iff (cs.IsLeftInversion (Inv.inv w) t) (cs.IsRightInversion w t)
  -/
  convert cs.isRightInversion_inv_iff.symm
  /-
    case h.e'_2.h.e'_6
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w t : W
    ⊢ Eq w (Inv.inv (Inv.inv w))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem isRightInversion_mul_left_iff {w : W} :
    cs.IsRightInversion (w * t) t ↔ ¬cs.IsRightInversion w t := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    t : W
    ht : cs.IsReflection t
    w : W
    ⊢ Iff (cs.IsRightInversion (HMul.hMul w t) t) (Not (cs.IsRightInversion w t))
  -/
  unfold IsRightInversion
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    t : W
    ht : cs.IsReflection t
    w : W
    ⊢ Iff (And (cs.IsReflection t) (LT.lt (cs.length (HMul.hMul (HMul.hMul w t) t) …
  -/
  simp only [mul_assoc, ht.inv, ht.mul_self, mul_one, ht, true_and, not_lt]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    t : W
    ht : cs.IsReflection t
    w : W
    ⊢ Iff (LT.lt (cs.length w) (cs.length (HMul.hMul w t))) (LE.le (cs.length w) ( …
  -/
  constructor
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      t : W
      ht : cs.IsReflection t
      w : W
      ⊢ LT.lt (cs.length w) (cs.length (HMul.hMul w t)) → LE.le (cs.length w) (cs.le …
    -/
  · exact le_of_lt
    /-
      🎉 no goals
    -/
    /-
      case mpr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      t : W
      ht : cs.IsReflection t
      w : W
      ⊢ LE.le (cs.length w) (cs.length (HMul.hMul w t)) → LT.lt (cs.length w) (cs.le …
    -/
  · exact (lt_of_le_of_ne' · (ht.length_mul_left_ne w))
    /-
      🎉 no goals
    -/


theorem not_isRightInversion_mul_left_iff {w : W} :
    ¬cs.IsRightInversion (w * t) t ↔ cs.IsRightInversion w t :=
  ht.isRightInversion_mul_left_iff.not_left


theorem isLeftInversion_mul_right_iff {w : W} :
    cs.IsLeftInversion (t * w) t ↔ ¬cs.IsLeftInversion w t := by
  rw [← isRightInversion_inv_iff, ← isRightInversion_inv_iff, mul_inv_rev, ht.inv,
    ht.isRightInversion_mul_left_iff]


theorem not_isLeftInversion_mul_right_iff {w : W}  :
    ¬cs.IsLeftInversion (t * w) t ↔ cs.IsLeftInversion w t :=
  ht.isLeftInversion_mul_right_iff.not_left


@[simp]
theorem isRightInversion_simple_iff_isRightDescent (w : W) (i : B) :
    cs.IsRightInversion w (s i) ↔ cs.IsRightDescent w i := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (cs.IsRightInversion w (cs.simple i)) (cs.IsRightDescent w i)
  -/
  simp [IsRightInversion, IsRightDescent, cs.isReflection_simple i]
  /-
    🎉 no goals
  -/


@[simp]
theorem isLeftInversion_simple_iff_isLeftDescent (w : W) (i : B) :
    cs.IsLeftInversion w (s i) ↔ cs.IsLeftDescent w i := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (cs.IsLeftInversion w (cs.simple i)) (cs.IsLeftDescent w i)
  -/
  simp [IsLeftInversion, IsLeftDescent, cs.isReflection_simple i]
  /-
    🎉 no goals
  -/


/-- The right inversion sequence of `ω`. The right inversion sequence of a word
$s_{i_1} \cdots s_{i_\ell}$ is the sequence
$$s_{i_\ell}\cdots s_{i_1}\cdots s_{i_\ell}, \ldots,
    s_{i_{\ell}}s_{i_{\ell - 1}}s_{i_{\ell - 2}}s_{i_{\ell - 1}}s_{i_\ell}, \ldots,
    s_{i_{\ell}}s_{i_{\ell - 1}}s_{i_\ell}, s_{i_\ell}.$$
-/
def rightInvSeq (ω : List B) : List W :=
  match ω with
  | []          => []
  | i :: ω      => (π ω)⁻¹ * (s i) * (π ω) :: rightInvSeq ω


/-- The left inversion sequence of `ω`. The left inversion sequence of a word
$s_{i_1} \cdots s_{i_\ell}$ is the sequence
$$s_{i_1}, s_{i_1}s_{i_2}s_{i_1}, s_{i_1}s_{i_2}s_{i_3}s_{i_2}s_{i_1}, \ldots,
    s_{i_1}\cdots s_{i_\ell}\cdots s_{i_1}.$$
-/
def leftInvSeq (ω : List B) : List W :=
  match ω with
  | []          => []
  | i :: ω      => s i :: List.map (MulAut.conj (s i)) (leftInvSeq ω)


local prefix:100 "ris" => cs.rightInvSeq

local prefix:100 "lis" => cs.leftInvSeq


@[simp] theorem rightInvSeq_nil : ris [] = [] := rfl


@[simp] theorem leftInvSeq_nil : lis [] = [] := rfl


                                                                      /-
                                                                        B : Type u_1
                                                                        W : Type u_2
                                                                        inst✝ : Group W
                                                                        M : CoxeterMatrix B
                                                                        cs : CoxeterSystem M W
                                                                        i : B
                                                                        ⊢ Eq (cs.rightInvSeq (List.cons i List.nil)) (List.cons (cs.simple i) List.nil)
                                                                      -/
@[simp] theorem rightInvSeq_singleton (i : B) : ris [i] = [s i] := by simp [rightInvSeq]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp] theorem leftInvSeq_singleton (i : B) : lis [i] = [s i] := rfl


theorem rightInvSeq_concat (ω : List B) (i : B) :
    ris (ω.concat i) = (List.map (MulAut.conj (s i)) (ris ω)).concat (s i) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    i : B
    ⊢ Eq (cs.rightInvSeq (ω.concat i)) ((List.map (⇑(MulAut.conj (cs.simple i))) ( …
  -/
  induction' ω with j ω ih
    /-
      case nil
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ⊢ Eq (cs.rightInvSeq (List.nil.concat i)) ((List.map (⇑(MulAut.conj (cs.simple …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i j : B
      ω : List B
      ih : Eq (cs.rightInvSeq (ω.concat i)) ((List.map (⇑(MulAut.conj (cs.simple i)) …
      ⊢ Eq (cs.rightInvSeq ((List.cons j ω).concat i)) ((List.map (⇑(MulAut.conj (cs …
    -/
  · dsimp [rightInvSeq, concat]
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i j : B
      ω : List B
      ih : Eq (cs.rightInvSeq (ω.concat i)) ((List.map (⇑(MulAut.conj (cs.simple i)) …
      ⊢ Eq (List.cons (HMul.hMul (HMul.hMul (Inv.inv (cs.wordProd (ω.concat i))) (cs …
    -/
    rw [ih]
    simp only [concat_eq_append, wordProd_append, wordProd_cons, wordProd_nil, mul_one, mul_inv_rev,
      inv_simple, cons_append, cons.injEq, and_true]
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i j : B
      ω : List B
      ih : Eq (cs.rightInvSeq (ω.concat i)) ((List.map (⇑(MulAut.conj (cs.simple i)) …
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (cs.simple i) (Inv.inv (cs.wordProd ω))) …
    -/
    group
    /-
      🎉 no goals
    -/


private theorem leftInvSeq_eq_reverse_rightInvSeq_reverse (ω : List B) :
    lis ω = (ris ω.reverse).reverse := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    ⊢ Eq (cs.leftInvSeq ω) (cs.rightInvSeq ω.reverse).reverse
  -/
  induction' ω with i ω ih
    /-
      case nil
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ⊢ Eq (cs.leftInvSeq List.nil) (cs.rightInvSeq List.nil.reverse).reverse
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ω : List B
      ih : Eq (cs.leftInvSeq ω) (cs.rightInvSeq ω.reverse).reverse
      ⊢ Eq (cs.leftInvSeq (List.cons i ω)) (cs.rightInvSeq (List.cons i ω).reverse). …
    -/
  · rw [leftInvSeq, reverse_cons, ← concat_eq_append, rightInvSeq_concat, ih]
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ω : List B
      ih : Eq (cs.leftInvSeq ω) (cs.rightInvSeq ω.reverse).reverse
      ⊢ Eq (List.cons (cs.simple i) (List.map (⇑(MulAut.conj (cs.simple i))) (cs.rig …
    -/
    simp [map_reverse]
    /-
      🎉 no goals
    -/


theorem leftInvSeq_concat (ω : List B) (i : B) :
    lis (ω.concat i) = (lis ω).concat ((π ω) * (s i) * (π ω)⁻¹) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    i : B
    ⊢ Eq (cs.leftInvSeq (ω.concat i)) ((cs.leftInvSeq ω).concat (HMul.hMul (HMul.h …
  -/
  simp [leftInvSeq_eq_reverse_rightInvSeq_reverse, rightInvSeq]
  /-
    🎉 no goals
  -/


theorem rightInvSeq_reverse (ω : List B) :
    ris (ω.reverse) = (lis ω).reverse := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    ⊢ Eq (cs.rightInvSeq ω.reverse) (cs.leftInvSeq ω).reverse
  -/
  simp [leftInvSeq_eq_reverse_rightInvSeq_reverse]
  /-
    🎉 no goals
  -/


theorem leftInvSeq_reverse (ω : List B) :
    lis (ω.reverse) = (ris ω).reverse := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    ⊢ Eq (cs.leftInvSeq ω.reverse) (cs.rightInvSeq ω).reverse
  -/
  simp [leftInvSeq_eq_reverse_rightInvSeq_reverse]
  /-
    🎉 no goals
  -/


@[simp] theorem length_rightInvSeq (ω : List B) : (ris ω).length = ω.length := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    ⊢ Eq (cs.rightInvSeq ω).length ω.length
  -/
  induction' ω with i ω ih
    /-
      case nil
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ⊢ Eq (cs.rightInvSeq List.nil).length List.nil.length
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ω : List B
      ih : Eq (cs.rightInvSeq ω).length ω.length
      ⊢ Eq (cs.rightInvSeq (List.cons i ω)).length (List.cons i ω).length
    -/
  · simpa [rightInvSeq]
    /-
      🎉 no goals
    -/


@[simp] theorem length_leftInvSeq (ω : List B) : (lis ω).length = ω.length := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    ⊢ Eq (cs.leftInvSeq ω).length ω.length
  -/
  simp [leftInvSeq_eq_reverse_rightInvSeq_reverse]
  /-
    🎉 no goals
  -/


theorem getD_rightInvSeq (ω : List B) (j : ℕ) :
    (ris ω).getD j 1 =
      (π (ω.drop (j + 1)))⁻¹
        * (Option.map (cs.simple) ω[j]?).getD 1
        * π (ω.drop (j + 1)) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq ((cs.rightInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (Inv.inv (cs.wordProd …
  -/
  induction' ω with i ω ih generalizing j
    /-
      case nil
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      j : Nat
      ⊢ Eq ((cs.rightInvSeq List.nil).getD j 1) (HMul.hMul (HMul.hMul (Inv.inv (cs.w …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ω : List B
      ih : ∀ (j : Nat), Eq ((cs.rightInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (Inv. …
      j : Nat
      ⊢ Eq ((cs.rightInvSeq (List.cons i ω)).getD j 1) (HMul.hMul (HMul.hMul (Inv.in …
    -/
  · dsimp only [rightInvSeq]
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ω : List B
      ih : ∀ (j : Nat), Eq ((cs.rightInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (Inv. …
      j : Nat
      ⊢ Eq ((List.cons (HMul.hMul (HMul.hMul (Inv.inv (cs.wordProd ω)) (cs.simple i) …
    -/
    rcases j with _ | j'
      /-
        case cons.zero
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i : B
        ω : List B
        ih : ∀ (j : Nat), Eq ((cs.rightInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (Inv. …
        ⊢ Eq ((List.cons (HMul.hMul (HMul.hMul (Inv.inv (cs.wordProd ω)) (cs.simple i) …
      -/
    · simp [getD_cons_zero]
      /-
        🎉 no goals
      -/
      /-
        case cons.succ
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i : B
        ω : List B
        ih : ∀ (j : Nat), Eq ((cs.rightInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (Inv. …
        j' : Nat
        ⊢ Eq ((List.cons (HMul.hMul (HMul.hMul (Inv.inv (cs.wordProd ω)) (cs.simple i) …
      -/
    · simp only [getD_eq_getElem?_getD, get?_eq_getElem?] at ih
      /-
        case cons.succ
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i : B
        ω : List B
        j' : Nat
        ih : ∀ (j : Nat), Eq ((GetElem?.getElem? (cs.rightInvSeq ω) j).getD 1) (HMul.h …
        ⊢ Eq ((List.cons (HMul.hMul (HMul.hMul (Inv.inv (cs.wordProd ω)) (cs.simple i) …
      -/
      simp [getD_cons_succ, ih j']
      /-
        🎉 no goals
      -/


lemma getElem_rightInvSeq (ω : List B) (j : ℕ) (h : j < ω.length) :
                   /-
                     B : Type u_1
                     W : Type u_2
                     inst✝ : Group W
                     M : CoxeterMatrix B
                     cs : CoxeterSystem M W
                     ω : List B
                     j : Nat
                     h : LT.lt j ω.length
                     ⊢ LT.lt j (cs.rightInvSeq ω).length
                   -/
    (ris ω)[j]'(by simp[h]) =
                   /-
                     🎉 no goals
                   -/
    (π (ω.drop (j + 1)))⁻¹
      * (Option.map (cs.simple) ω[j]?).getD 1
      * π (ω.drop (j + 1)) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    h : LT.lt j ω.length
    ⊢ Eq (GetElem.getElem (cs.rightInvSeq ω) j ⋯) (HMul.hMul (HMul.hMul (Inv.inv ( …
  -/
  rw [← List.getD_eq_getElem (ris ω) 1, getD_rightInvSeq]
  /-
    🎉 no goals
  -/


theorem getD_leftInvSeq (ω : List B) (j : ℕ) :
    (lis ω).getD j 1 =
      π (ω.take j)
        * (Option.map (cs.simple) ω[j]?).getD 1
        * (π (ω.take j))⁻¹ := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq ((cs.leftInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (cs.wordProd (List.tak …
  -/
  induction' ω with i ω ih generalizing j
    /-
      case nil
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      j : Nat
      ⊢ Eq ((cs.leftInvSeq List.nil).getD j 1) (HMul.hMul (HMul.hMul (cs.wordProd (L …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ω : List B
      ih : ∀ (j : Nat), Eq ((cs.leftInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (cs.wo …
      j : Nat
      ⊢ Eq ((cs.leftInvSeq (List.cons i ω)).getD j 1) (HMul.hMul (HMul.hMul (cs.word …
    -/
  · dsimp [leftInvSeq]
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ω : List B
      ih : ∀ (j : Nat), Eq ((cs.leftInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (cs.wo …
      j : Nat
      ⊢ Eq ((List.cons (cs.simple i) (List.map (⇑(MulAut.conj (cs.simple i))) (cs.le …
    -/
    rcases j with _ | j'
      /-
        case cons.zero
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i : B
        ω : List B
        ih : ∀ (j : Nat), Eq ((cs.leftInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (cs.wo …
        ⊢ Eq ((List.cons (cs.simple i) (List.map (⇑(MulAut.conj (cs.simple i))) (cs.le …
      -/
    · simp [getD_cons_zero]
      /-
        🎉 no goals
      -/
      /-
        case cons.succ
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i : B
        ω : List B
        ih : ∀ (j : Nat), Eq ((cs.leftInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (cs.wo …
        j' : Nat
        ⊢ Eq ((List.cons (cs.simple i) (List.map (⇑(MulAut.conj (cs.simple i))) (cs.le …
      -/
    · rw [getD_cons_succ]
      /-
        case cons.succ
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i : B
        ω : List B
        ih : ∀ (j : Nat), Eq ((cs.leftInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (cs.wo …
        j' : Nat
        ⊢ Eq ((List.map (⇑(MulAut.conj (cs.simple i))) (cs.leftInvSeq ω)).getD j' 1) ( …
      -/
      rw [(by simp : 1 = ⇑(MulAut.conj (s i)) 1)]
      /-
        case cons.succ
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i : B
        ω : List B
        ih : ∀ (j : Nat), Eq ((cs.leftInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (cs.wo …
        j' : Nat
        ⊢ Eq ((List.map (⇑(MulAut.conj (cs.simple i))) (cs.leftInvSeq ω)).getD j' ((Mu …
      -/
      rw [getD_map]
      /-
        case cons.succ
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i : B
        ω : List B
        ih : ∀ (j : Nat), Eq ((cs.leftInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (cs.wo …
        j' : Nat
        ⊢ Eq ((MulAut.conj (cs.simple i)) ((cs.leftInvSeq ω).getD j' 1)) (HMul.hMul (H …
      -/
      rw [ih j']
      /-
        case cons.succ
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i : B
        ω : List B
        ih : ∀ (j : Nat), Eq ((cs.leftInvSeq ω).getD j 1) (HMul.hMul (HMul.hMul (cs.wo …
        j' : Nat
        ⊢ Eq ((MulAut.conj (cs.simple i)) (HMul.hMul (HMul.hMul (cs.wordProd (List.tak …
      -/
      simp [← mul_assoc, wordProd_cons]
      /-
        🎉 no goals
      -/


lemma getElem_leftInvSeq (ω : List B) (j : ℕ) (h : j < ω.length) :
                   /-
                     B : Type u_1
                     W : Type u_2
                     inst✝ : Group W
                     M : CoxeterMatrix B
                     cs : CoxeterSystem M W
                     ω : List B
                     j : Nat
                     h : LT.lt j ω.length
                     ⊢ LT.lt j (cs.leftInvSeq ω).length
                   -/
    (lis ω)[j]'(by simp[h]) =
                   /-
                     🎉 no goals
                   -/
    cs.wordProd (List.take j ω) * s ω[j] * (cs.wordProd (List.take j ω))⁻¹ := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    h : LT.lt j ω.length
    ⊢ Eq (GetElem.getElem (cs.leftInvSeq ω) j ⋯) (HMul.hMul (HMul.hMul (cs.wordPro …
  -/
  rw [← List.getD_eq_getElem (lis ω) 1, getD_leftInvSeq]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    h : LT.lt j ω.length
    ⊢ Eq (HMul.hMul (HMul.hMul (cs.wordProd (List.take j ω)) ((Option.map cs.simpl …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


theorem getD_rightInvSeq_mul_self (ω : List B) (j : ℕ) :
    ((ris ω).getD j 1) * ((ris ω).getD j 1) = 1 := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul ((cs.rightInvSeq ω).getD j 1) ((cs.rightInvSeq ω).getD j 1)) 1
  -/
  simp_rw [getD_rightInvSeq, mul_assoc]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul (Inv.inv (cs.wordProd (List.drop (HAdd.hAdd j 1) ω))) (HMul.hM …
  -/
  rcases em (j < ω.length) with hj | nhj
    /-
      case inl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      hj : LT.lt j ω.length
      ⊢ Eq (HMul.hMul (Inv.inv (cs.wordProd (List.drop (HAdd.hAdd j 1) ω))) (HMul.hM …
    -/
  · rw [getElem?_eq_getElem hj]
    /-
      case inl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      hj : LT.lt j ω.length
      ⊢ Eq (HMul.hMul (Inv.inv (cs.wordProd (List.drop (HAdd.hAdd j 1) ω))) (HMul.hM …
    -/
    simp [← mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case inr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      nhj : Not (LT.lt j ω.length)
      ⊢ Eq (HMul.hMul (Inv.inv (cs.wordProd (List.drop (HAdd.hAdd j 1) ω))) (HMul.hM …
    -/
  · rw [getElem?_eq_none_iff.mpr (by omega)]
    /-
      case inr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      nhj : Not (LT.lt j ω.length)
      ⊢ Eq (HMul.hMul (Inv.inv (cs.wordProd (List.drop (HAdd.hAdd j 1) ω))) (HMul.hM …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem getD_leftInvSeq_mul_self (ω : List B) (j : ℕ) :
    ((lis ω).getD j 1) * ((lis ω).getD j 1) = 1 := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul ((cs.leftInvSeq ω).getD j 1) ((cs.leftInvSeq ω).getD j 1)) 1
  -/
  simp_rw [getD_leftInvSeq, mul_assoc]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul (cs.wordProd (List.take j ω)) (HMul.hMul ((Option.map cs.simpl …
  -/
  rcases em (j < ω.length) with hj | nhj
    /-
      case inl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      hj : LT.lt j ω.length
      ⊢ Eq (HMul.hMul (cs.wordProd (List.take j ω)) (HMul.hMul ((Option.map cs.simpl …
    -/
  · rw [getElem?_eq_getElem hj]
    /-
      case inl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      hj : LT.lt j ω.length
      ⊢ Eq (HMul.hMul (cs.wordProd (List.take j ω)) (HMul.hMul ((Option.map cs.simpl …
    -/
    simp [← mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case inr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      nhj : Not (LT.lt j ω.length)
      ⊢ Eq (HMul.hMul (cs.wordProd (List.take j ω)) (HMul.hMul ((Option.map cs.simpl …
    -/
  · rw [getElem?_eq_none_iff.mpr (by omega)]
    /-
      case inr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      nhj : Not (LT.lt j ω.length)
      ⊢ Eq (HMul.hMul (cs.wordProd (List.take j ω)) (HMul.hMul ((Option.map cs.simpl …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem rightInvSeq_drop (ω : List B) (j : ℕ) :
    ris (ω.drop j) = (ris ω).drop j := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (cs.rightInvSeq (List.drop j ω)) (List.drop j (cs.rightInvSeq ω))
  -/
  induction' j with j ih₁ generalizing ω
    /-
      case zero
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      ⊢ Eq (cs.rightInvSeq (List.drop 0 ω)) (List.drop 0 (cs.rightInvSeq ω))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      j : Nat
      ih₁ : ∀ (ω : List B), Eq (cs.rightInvSeq (List.drop j ω)) (List.drop j (cs.rig …
      ω : List B
      ⊢ Eq (cs.rightInvSeq (List.drop (HAdd.hAdd j 1) ω)) (List.drop (HAdd.hAdd j 1) …
    -/
  · induction' ω with k ω _
      /-
        case succ.nil
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        j : Nat
        ih₁ : ∀ (ω : List B), Eq (cs.rightInvSeq (List.drop j ω)) (List.drop j (cs.rig …
        ⊢ Eq (cs.rightInvSeq (List.drop (HAdd.hAdd j 1) List.nil)) (List.drop (HAdd.hA …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case succ.cons
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        j : Nat
        ih₁ : ∀ (ω : List B), Eq (cs.rightInvSeq (List.drop j ω)) (List.drop j (cs.rig …
        k : B
        ω : List B
        tail_ih✝ : Eq (cs.rightInvSeq (List.drop (HAdd.hAdd j 1) ω)) (List.drop (HAdd. …
        ⊢ Eq (cs.rightInvSeq (List.drop (HAdd.hAdd j 1) (List.cons k ω))) (List.drop ( …
      -/
    · rw [drop_succ_cons, ih₁ ω, rightInvSeq, drop_succ_cons]
      /-
        🎉 no goals
      -/


theorem leftInvSeq_take (ω : List B) (j : ℕ) :
    lis (ω.take j) = (lis ω).take j := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (cs.leftInvSeq (List.take j ω)) (List.take j (cs.leftInvSeq ω))
  -/
  simp only [leftInvSeq_eq_reverse_rightInvSeq_reverse]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (cs.rightInvSeq (List.take j ω).reverse).reverse (List.take j (cs.rightIn …
  -/
  rw [List.take_reverse]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (cs.rightInvSeq (List.take j ω).reverse).reverse (List.drop (HSub.hSub (c …
  -/
  nth_rw 1 [← List.reverse_reverse ω]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (cs.rightInvSeq (List.take j ω.reverse.reverse).reverse).reverse (List.dr …
  -/
  rw [List.take_reverse]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (cs.rightInvSeq (List.drop (HSub.hSub ω.reverse.length j) ω.reverse).reve …
  -/
  simp [rightInvSeq_drop]
  /-
    🎉 no goals
  -/


theorem isReflection_of_mem_rightInvSeq (ω : List B) {t : W} (ht : t ∈ ris ω) :
    cs.IsReflection t := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    t : W
    ht : Membership.mem (cs.rightInvSeq ω) t
    ⊢ cs.IsReflection t
  -/
  induction' ω with i ω ih
    /-
      case nil
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      t : W
      ht : Membership.mem (cs.rightInvSeq List.nil) t
      ⊢ cs.IsReflection t
    -/
  · simp at ht
    /-
      🎉 no goals
    -/
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      t : W
      i : B
      ω : List B
      ih : Membership.mem (cs.rightInvSeq ω) t → cs.IsReflection t
      ht : Membership.mem (cs.rightInvSeq (List.cons i ω)) t
      ⊢ cs.IsReflection t
    -/
  · dsimp [rightInvSeq] at ht
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      t : W
      i : B
      ω : List B
      ih : Membership.mem (cs.rightInvSeq ω) t → cs.IsReflection t
      ht : Membership.mem (List.cons (HMul.hMul (HMul.hMul (Inv.inv (cs.wordProd ω)) …
      ⊢ cs.IsReflection t
    -/
    rcases ht with _ | ⟨_, mem⟩
      /-
        case cons.head
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i : B
        ω : List B
        ih : Membership.mem (cs.rightInvSeq ω) (HMul.hMul (HMul.hMul (Inv.inv (cs.word …
        ⊢ cs.IsReflection (HMul.hMul (HMul.hMul (Inv.inv (cs.wordProd ω)) (cs.simple i …
      -/
    · use (π ω)⁻¹, i
      /-
        case h
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i : B
        ω : List B
        ih : Membership.mem (cs.rightInvSeq ω) (HMul.hMul (HMul.hMul (Inv.inv (cs.word …
        ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv (cs.wordProd ω)) (cs.simple i)) (cs.wordPr …
      -/
      group
      /-
        🎉 no goals
      -/
      /-
        case cons.tail
        B : Type u_1
        W : Type u_2
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        t : W
        i : B
        ω : List B
        ih : Membership.mem (cs.rightInvSeq ω) t → cs.IsReflection t
        mem : List.Mem t (cs.rightInvSeq ω)
        ⊢ cs.IsReflection t
      -/
    · exact ih mem
      /-
        🎉 no goals
      -/


theorem isReflection_of_mem_leftInvSeq (ω : List B) {t : W} (ht : t ∈ lis ω) :
    cs.IsReflection t := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    t : W
    ht : Membership.mem (cs.leftInvSeq ω) t
    ⊢ cs.IsReflection t
  -/
  simp only [leftInvSeq_eq_reverse_rightInvSeq_reverse, mem_reverse] at ht
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    t : W
    ht : Membership.mem (cs.rightInvSeq ω.reverse) t
    ⊢ cs.IsReflection t
  -/
  exact cs.isReflection_of_mem_rightInvSeq ω.reverse ht
  /-
    🎉 no goals
  -/


theorem wordProd_mul_getD_rightInvSeq (ω : List B) (j : ℕ) :
    π ω * ((ris ω).getD j 1) = π (ω.eraseIdx j) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul (cs.wordProd ω) ((cs.rightInvSeq ω).getD j 1)) (cs.wordProd (ω …
  -/
  rw [getD_rightInvSeq, eraseIdx_eq_take_drop_succ]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul (cs.wordProd ω) (HMul.hMul (HMul.hMul (Inv.inv (cs.wordProd (L …
  -/
  nth_rw 1 [← take_append_drop (j + 1) ω]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul (cs.wordProd (HAppend.hAppend (List.take (HAdd.hAdd j 1) ω) (L …
  -/
  rw [take_succ]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul (cs.wordProd (HAppend.hAppend (HAppend.hAppend (List.take j ω) …
  -/
  obtain lt | le := lt_or_le j ω.length
    /-
      case inl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      lt : LT.lt j ω.length
      ⊢ Eq (HMul.hMul (cs.wordProd (HAppend.hAppend (HAppend.hAppend (List.take j ω) …
    -/
  · simp only [get?_eq_getElem?, getElem?_eq_getElem lt, wordProd_append, wordProd_cons, mul_assoc]
    /-
      case inl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      lt : LT.lt j ω.length
      ⊢ Eq (HMul.hMul (cs.wordProd (List.take j ω)) (HMul.hMul (cs.wordProd (Option. …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      le : LE.le ω.length j
      ⊢ Eq (HMul.hMul (cs.wordProd (HAppend.hAppend (HAppend.hAppend (List.take j ω) …
    -/
  · simp only [get?_eq_getElem?, getElem?_eq_none le]
    /-
      case inr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      le : LE.le ω.length j
      ⊢ Eq (HMul.hMul (cs.wordProd (HAppend.hAppend (HAppend.hAppend (List.take j ω) …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem getD_leftInvSeq_mul_wordProd (ω : List B) (j : ℕ) :
    ((lis ω).getD j 1) * π ω = π (ω.eraseIdx j) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul ((cs.leftInvSeq ω).getD j 1) (cs.wordProd ω)) (cs.wordProd (ω. …
  -/
  rw [getD_leftInvSeq, eraseIdx_eq_take_drop_succ]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (cs.wordProd (List.take j ω)) ((Option.m …
  -/
  nth_rw 4 [← take_append_drop (j + 1) ω]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (cs.wordProd (List.take j ω)) ((Option.m …
  -/
  rw [take_succ]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    j : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (cs.wordProd (List.take j ω)) ((Option.m …
  -/
  obtain lt | le := lt_or_le j ω.length
    /-
      case inl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      lt : LT.lt j ω.length
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (cs.wordProd (List.take j ω)) ((Option.m …
    -/
  · simp only [get?_eq_getElem?, getElem?_eq_getElem lt, wordProd_append, wordProd_cons, mul_assoc]
    /-
      case inl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      lt : LT.lt j ω.length
      ⊢ Eq (HMul.hMul (cs.wordProd (List.take j ω)) (HMul.hMul ((Option.map cs.simpl …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      le : LE.le ω.length j
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (cs.wordProd (List.take j ω)) ((Option.m …
    -/
  · simp only [get?_eq_getElem?, getElem?_eq_none le]
    /-
      case inr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      j : Nat
      le : LE.le ω.length j
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (cs.wordProd (List.take j ω)) ((Option.m …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem isRightInversion_of_mem_rightInvSeq {ω : List B} (hω : cs.IsReduced ω) {t : W}
    (ht : t ∈ ris ω) : cs.IsRightInversion (π ω) t := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    hω : cs.IsReduced ω
    t : W
    ht : Membership.mem (cs.rightInvSeq ω) t
    ⊢ cs.IsRightInversion (cs.wordProd ω) t
  -/
  constructor
    /-
      case left
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      hω : cs.IsReduced ω
      t : W
      ht : Membership.mem (cs.rightInvSeq ω) t
      ⊢ cs.IsReflection t
    -/
  · exact cs.isReflection_of_mem_rightInvSeq ω ht
    /-
      🎉 no goals
    -/
    /-
      case right
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      hω : cs.IsReduced ω
      t : W
      ht : Membership.mem (cs.rightInvSeq ω) t
      ⊢ LT.lt (cs.length (HMul.hMul (cs.wordProd ω) t)) (cs.length (cs.wordProd ω))
    -/
  · obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.mp ht
    /-
      case right.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      hω : cs.IsReduced ω
      j : Nat
      hj : LT.lt j (cs.rightInvSeq ω).length
      ht : Membership.mem (cs.rightInvSeq ω) (GetElem.getElem (cs.rightInvSeq ω) j hj)
      ⊢ LT.lt (cs.length (HMul.hMul (cs.wordProd ω) (GetElem.getElem (cs.rightInvSeq …
    -/
    rw [← List.getD_eq_getElem _ 1 hj, wordProd_mul_getD_rightInvSeq]
    /-
      case right.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      hω : cs.IsReduced ω
      j : Nat
      hj : LT.lt j (cs.rightInvSeq ω).length
      ht : Membership.mem (cs.rightInvSeq ω) (GetElem.getElem (cs.rightInvSeq ω) j hj)
      ⊢ LT.lt (cs.length (cs.wordProd (ω.eraseIdx j))) (cs.length (cs.wordProd ω))
    -/
    rw [cs.length_rightInvSeq] at hj
    calc
      ℓ (π (ω.eraseIdx j))
      _ ≤ (ω.eraseIdx j).length   := cs.length_wordProd_le _
      _ < ω.length                := by rw [← List.length_eraseIdx_add_one hj]; exact lt_add_one _
      _ = ℓ (π ω)                 := hω.symm


theorem isLeftInversion_of_mem_leftInvSeq {ω : List B} (hω : cs.IsReduced ω) {t : W}
    (ht : t ∈ lis ω) : cs.IsLeftInversion (π ω) t := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    hω : cs.IsReduced ω
    t : W
    ht : Membership.mem (cs.leftInvSeq ω) t
    ⊢ cs.IsLeftInversion (cs.wordProd ω) t
  -/
  constructor
    /-
      case left
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      hω : cs.IsReduced ω
      t : W
      ht : Membership.mem (cs.leftInvSeq ω) t
      ⊢ cs.IsReflection t
    -/
  · exact cs.isReflection_of_mem_leftInvSeq ω ht
    /-
      🎉 no goals
    -/
    /-
      case right
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      hω : cs.IsReduced ω
      t : W
      ht : Membership.mem (cs.leftInvSeq ω) t
      ⊢ LT.lt (cs.length (HMul.hMul t (cs.wordProd ω))) (cs.length (cs.wordProd ω))
    -/
  · obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.mp ht
    /-
      case right.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      hω : cs.IsReduced ω
      j : Nat
      hj : LT.lt j (cs.leftInvSeq ω).length
      ht : Membership.mem (cs.leftInvSeq ω) (GetElem.getElem (cs.leftInvSeq ω) j hj)
      ⊢ LT.lt (cs.length (HMul.hMul (GetElem.getElem (cs.leftInvSeq ω) j hj) (cs.wor …
    -/
    rw [← List.getD_eq_getElem _ 1 hj, getD_leftInvSeq_mul_wordProd]
    /-
      case right.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      hω : cs.IsReduced ω
      j : Nat
      hj : LT.lt j (cs.leftInvSeq ω).length
      ht : Membership.mem (cs.leftInvSeq ω) (GetElem.getElem (cs.leftInvSeq ω) j hj)
      ⊢ LT.lt (cs.length (cs.wordProd (ω.eraseIdx j))) (cs.length (cs.wordProd ω))
    -/
    rw [cs.length_leftInvSeq] at hj
    calc
      ℓ (π (ω.eraseIdx j))
      _ ≤ (ω.eraseIdx j).length   := cs.length_wordProd_le _
      _ < ω.length                := by rw [← List.length_eraseIdx_add_one hj]; exact lt_add_one _
      _ = ℓ (π ω)                 := hω.symm


theorem prod_rightInvSeq (ω : List B) : prod (ris ω) = (π ω)⁻¹ := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    ⊢ Eq (cs.rightInvSeq ω).prod (Inv.inv (cs.wordProd ω))
  -/
  induction' ω with i ω ih
    /-
      case nil
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ⊢ Eq (cs.rightInvSeq List.nil).prod (Inv.inv (cs.wordProd List.nil))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ω : List B
      ih : Eq (cs.rightInvSeq ω).prod (Inv.inv (cs.wordProd ω))
      ⊢ Eq (cs.rightInvSeq (List.cons i ω)).prod (Inv.inv (cs.wordProd (List.cons i  …
    -/
  · simp [rightInvSeq, ih, wordProd_cons]
    /-
      🎉 no goals
    -/


theorem prod_leftInvSeq (ω : List B) : prod (lis ω) = (π ω)⁻¹ := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    ⊢ Eq (cs.leftInvSeq ω).prod (Inv.inv (cs.wordProd ω))
  -/
  simp only [leftInvSeq_eq_reverse_rightInvSeq_reverse, prod_reverse_noncomm, inv_inj]
  have : List.map (fun x ↦ x⁻¹) (ris ω.reverse) = ris ω.reverse := calc
    List.map (fun x ↦ x⁻¹) (ris ω.reverse)
    _ = List.map id (ris ω.reverse)             := by
        apply List.map_congr_left
        intro t ht
        exact (cs.isReflection_of_mem_rightInvSeq _ ht).inv
    _ = ris ω.reverse                           := map_id _
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    this : Eq (List.map (fun x => Inv.inv x) (cs.rightInvSeq ω.reverse)) (cs.right …
    ⊢ Eq (List.map (fun x => Inv.inv x) (cs.rightInvSeq ω.reverse)).prod (cs.wordP …
  -/
  rw [this]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    this : Eq (List.map (fun x => Inv.inv x) (cs.rightInvSeq ω.reverse)) (cs.right …
    ⊢ Eq (cs.rightInvSeq ω.reverse).prod (cs.wordProd ω)
  -/
  nth_rw 2 [← reverse_reverse ω]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    this : Eq (List.map (fun x => Inv.inv x) (cs.rightInvSeq ω.reverse)) (cs.right …
    ⊢ Eq (cs.rightInvSeq ω.reverse).prod (cs.wordProd ω.reverse.reverse)
  -/
  rw [wordProd_reverse]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    this : Eq (List.map (fun x => Inv.inv x) (cs.rightInvSeq ω.reverse)) (cs.right …
    ⊢ Eq (cs.rightInvSeq ω.reverse).prod (Inv.inv (cs.wordProd ω.reverse))
  -/
  exact cs.prod_rightInvSeq _
  /-
    🎉 no goals
  -/


theorem IsReduced.nodup_rightInvSeq {ω : List B} (rω : cs.IsReduced ω) : List.Nodup (ris ω) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    ⊢ (cs.rightInvSeq ω).Nodup
  -/
  apply List.nodup_iff_getElem?_ne_getElem?.mpr
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    ⊢ ∀ (i j : Nat), LT.lt i j → LT.lt j (cs.rightInvSeq ω).length → Ne (GetElem?. …
  -/
  intro j j' j_lt_j' j'_lt_length (dup : (rightInvSeq cs ω)[j]? = (rightInvSeq cs ω)[j']?)
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    j j' : Nat
    j_lt_j' : LT.lt j j'
    j'_lt_length : LT.lt j' (cs.rightInvSeq ω).length
    dup : Eq (GetElem?.getElem? (cs.rightInvSeq ω) j) (GetElem?.getElem? (cs.right …
    ⊢ False
  -/
  show False
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    j j' : Nat
    j_lt_j' : LT.lt j j'
    j'_lt_length : LT.lt j' (cs.rightInvSeq ω).length
    dup : Eq (GetElem?.getElem? (cs.rightInvSeq ω) j) (GetElem?.getElem? (cs.right …
    ⊢ False
  -/
  replace j'_lt_length : j' < List.length ω := by simpa using j'_lt_length
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    j j' : Nat
    j_lt_j' : LT.lt j j'
    dup : Eq (GetElem?.getElem? (cs.rightInvSeq ω) j) (GetElem?.getElem? (cs.right …
    j'_lt_length : LT.lt j' ω.length
    ⊢ False
  -/
  rw [getElem?_eq_getElem (by simp; omega), getElem?_eq_getElem (by simp; omega)] at dup
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    j j' : Nat
    j_lt_j' : LT.lt j j'
    j'_lt_length : LT.lt j' ω.length
    dup : Eq (Option.some (GetElem.getElem (cs.rightInvSeq ω) j ⋯)) (Option.some ( …
    ⊢ False
  -/
  apply Option.some_injective at dup
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    j j' : Nat
    j_lt_j' : LT.lt j j'
    j'_lt_length : LT.lt j' ω.length
    dup : Eq (GetElem.getElem (cs.rightInvSeq ω) j ⋯) (GetElem.getElem (cs.rightIn …
    ⊢ False
  -/
  rw [← getD_eq_getElem _ 1, ← getD_eq_getElem _ 1] at dup
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    j j' : Nat
    j_lt_j' : LT.lt j j'
    j'_lt_length : LT.lt j' ω.length
    dup : Eq ((cs.rightInvSeq ω).getD j 1) ((cs.rightInvSeq ω).getD j' 1)
    ⊢ False
  -/
  set! t := (ris ω).getD j 1 with h₁
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    j j' : Nat
    j_lt_j' : LT.lt j j'
    j'_lt_length : LT.lt j' ω.length
    dup : Eq ((cs.rightInvSeq ω).getD j 1) ((cs.rightInvSeq ω).getD j' 1)
    t : W := (cs.rightInvSeq ω).getD j 1
    h₁ : Eq t ((cs.rightInvSeq ω).getD j 1)
    ⊢ False
  -/
  set! t' := (ris (ω.eraseIdx j)).getD (j' - 1) 1 with h₂
  have h₃ : t' = (ris ω).getD j' 1                    := by
    rw [h₂, cs.getD_rightInvSeq, cs.getD_rightInvSeq,
      (Nat.sub_add_cancel (by omega) : j' - 1 + 1 = j'), eraseIdx_eq_take_drop_succ,
      drop_append_eq_append_drop, drop_of_length_le (by simp [j_lt_j'.le]), length_take,
      drop_drop, nil_append, min_eq_left_of_lt (j_lt_j'.trans j'_lt_length), Nat.add_comm,
      ← add_assoc, Nat.sub_add_cancel (by omega), mul_left_inj, mul_right_inj]
    congr 2
    show (List.take j ω ++ List.drop (j + 1) ω)[j' - 1]? = ω[j']?
    rw [getElem?_append_right (by simp [Nat.le_sub_one_of_lt j_lt_j']), getElem?_drop]
    congr
    show j + 1 + (j' - 1 - List.length (List.take j ω)) = j'
    rw [length_take]
    omega
  have h₄ : t * t' = 1                                := by
    rw [h₁, h₃, dup]
    exact cs.getD_rightInvSeq_mul_self _ _
  have h₅ := calc
    π ω   = π ω * t * t'                              := by rw [mul_assoc, h₄]; group
    _     = (π (ω.eraseIdx j)) * t'                   :=
        congrArg (· * t') (cs.wordProd_mul_getD_rightInvSeq _ _)
    _     = π ((ω.eraseIdx j).eraseIdx (j' - 1))      :=
        cs.wordProd_mul_getD_rightInvSeq _ _
  have h₆ := calc
    ω.length = ℓ (π ω)                                    := rω.symm
    _        = ℓ (π ((ω.eraseIdx j).eraseIdx (j' - 1)))   := congrArg cs.length h₅
    _        ≤ ((ω.eraseIdx j).eraseIdx (j' - 1)).length  := cs.length_wordProd_le _
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    j j' : Nat
    j_lt_j' : LT.lt j j'
    j'_lt_length : LT.lt j' ω.length
    dup : Eq ((cs.rightInvSeq ω).getD j 1) ((cs.rightInvSeq ω).getD j' 1)
    t : W := (cs.rightInvSeq ω).getD j 1
    h₁ : Eq t ((cs.rightInvSeq ω).getD j 1)
    t' : W := (cs.rightInvSeq (ω.eraseIdx j)).getD (HSub.hSub j' 1) 1
    h₂ : Eq t' ((cs.rightInvSeq (ω.eraseIdx j)).getD (HSub.hSub j' 1) 1)
    h₃ : Eq t' ((cs.rightInvSeq ω).getD j' 1)
    h₄ : Eq (HMul.hMul t t') 1
    h₅ : Eq (cs.wordProd ω) (cs.wordProd ((ω.eraseIdx j).eraseIdx (HSub.hSub j' 1)))
    h₆ : LE.le ω.length ((ω.eraseIdx j).eraseIdx (HSub.hSub j' 1)).length
    ⊢ False
  -/
  have h₇ := add_le_add_right (add_le_add_right h₆ 1) 1
  have h₈ : j' - 1 < List.length (eraseIdx ω j)           := by
    apply (@Nat.add_lt_add_iff_right 1).mp
    rw [Nat.sub_add_cancel (by omega)]
    rw [length_eraseIdx_add_one (by omega)]
    omega
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    j j' : Nat
    j_lt_j' : LT.lt j j'
    j'_lt_length : LT.lt j' ω.length
    dup : Eq ((cs.rightInvSeq ω).getD j 1) ((cs.rightInvSeq ω).getD j' 1)
    t : W := (cs.rightInvSeq ω).getD j 1
    h₁ : Eq t ((cs.rightInvSeq ω).getD j 1)
    t' : W := (cs.rightInvSeq (ω.eraseIdx j)).getD (HSub.hSub j' 1) 1
    h₂ : Eq t' ((cs.rightInvSeq (ω.eraseIdx j)).getD (HSub.hSub j' 1) 1)
    h₃ : Eq t' ((cs.rightInvSeq ω).getD j' 1)
    h₄ : Eq (HMul.hMul t t') 1
    h₅ : Eq (cs.wordProd ω) (cs.wordProd ((ω.eraseIdx j).eraseIdx (HSub.hSub j' 1)))
    h₆ : LE.le ω.length ((ω.eraseIdx j).eraseIdx (HSub.hSub j' 1)).length
    h₇ : LE.le (HAdd.hAdd (HAdd.hAdd ω.length 1) 1) (HAdd.hAdd (HAdd.hAdd ((ω.eras …
    h₈ : LT.lt (HSub.hSub j' 1) (ω.eraseIdx j).length
    ⊢ False
  -/
  rw [length_eraseIdx_add_one h₈] at h₇
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    j j' : Nat
    j_lt_j' : LT.lt j j'
    j'_lt_length : LT.lt j' ω.length
    dup : Eq ((cs.rightInvSeq ω).getD j 1) ((cs.rightInvSeq ω).getD j' 1)
    t : W := (cs.rightInvSeq ω).getD j 1
    h₁ : Eq t ((cs.rightInvSeq ω).getD j 1)
    t' : W := (cs.rightInvSeq (ω.eraseIdx j)).getD (HSub.hSub j' 1) 1
    h₂ : Eq t' ((cs.rightInvSeq (ω.eraseIdx j)).getD (HSub.hSub j' 1) 1)
    h₃ : Eq t' ((cs.rightInvSeq ω).getD j' 1)
    h₄ : Eq (HMul.hMul t t') 1
    h₅ : Eq (cs.wordProd ω) (cs.wordProd ((ω.eraseIdx j).eraseIdx (HSub.hSub j' 1)))
    h₆ : LE.le ω.length ((ω.eraseIdx j).eraseIdx (HSub.hSub j' 1)).length
    h₇ : LE.le (HAdd.hAdd (HAdd.hAdd ω.length 1) 1) (HAdd.hAdd (ω.eraseIdx j).leng …
    h₈ : LT.lt (HSub.hSub j' 1) (ω.eraseIdx j).length
    ⊢ False
  -/
  rw [length_eraseIdx_add_one (by omega)] at h₇
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    j j' : Nat
    j_lt_j' : LT.lt j j'
    j'_lt_length : LT.lt j' ω.length
    dup : Eq ((cs.rightInvSeq ω).getD j 1) ((cs.rightInvSeq ω).getD j' 1)
    t : W := (cs.rightInvSeq ω).getD j 1
    h₁ : Eq t ((cs.rightInvSeq ω).getD j 1)
    t' : W := (cs.rightInvSeq (ω.eraseIdx j)).getD (HSub.hSub j' 1) 1
    h₂ : Eq t' ((cs.rightInvSeq (ω.eraseIdx j)).getD (HSub.hSub j' 1) 1)
    h₃ : Eq t' ((cs.rightInvSeq ω).getD j' 1)
    h₄ : Eq (HMul.hMul t t') 1
    h₅ : Eq (cs.wordProd ω) (cs.wordProd ((ω.eraseIdx j).eraseIdx (HSub.hSub j' 1)))
    h₆ : LE.le ω.length ((ω.eraseIdx j).eraseIdx (HSub.hSub j' 1)).length
    h₇ : LE.le (HAdd.hAdd (HAdd.hAdd ω.length 1) 1) ω.length
    h₈ : LT.lt (HSub.hSub j' 1) (ω.eraseIdx j).length
    ⊢ False
  -/
  omega
  /-
    🎉 no goals
  -/


theorem IsReduced.nodup_leftInvSeq {ω : List B} (rω : cs.IsReduced ω) : List.Nodup (lis ω) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    ⊢ (cs.leftInvSeq ω).Nodup
  -/
  simp only [leftInvSeq_eq_reverse_rightInvSeq_reverse, nodup_reverse]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    ⊢ (cs.rightInvSeq ω.reverse).Nodup
  -/
  apply nodup_rightInvSeq
  /-
    case rω
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    rω : cs.IsReduced ω
    ⊢ cs.IsReduced ω.reverse
  -/
  rwa [isReduced_reverse_iff]
  /-
    🎉 no goals
  -/


lemma getElem_succ_leftInvSeq_alternatingWord
    (i j : B) (p k : ℕ) (h : k + 1 < 2 * p) :
                                                   /-
                                                     B : Type u_1
                                                     W : Type u_2
                                                     inst✝ : Group W
                                                     M : CoxeterMatrix B
                                                     cs : CoxeterSystem M W
                                                     i j : B
                                                     p k : Nat
                                                     h : LT.lt (HAdd.hAdd k 1) (HMul.hMul 2 p)
                                                     ⊢ LT.lt (HAdd.hAdd k 1) (cs.leftInvSeq (CoxeterSystem.alternatingWord i j (HMu …
                                                   -/
    (lis (alternatingWord i j (2 * p)))[k + 1]'(by simp; exact h) =
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                                  /-
                                                                    B : Type u_1
                                                                    W : Type u_2
                                                                    inst✝ : Group W
                                                                    M : CoxeterMatrix B
                                                                    cs : CoxeterSystem M W
                                                                    i j : B
                                                                    p k : Nat
                                                                    h : LT.lt (HAdd.hAdd k 1) (HMul.hMul 2 p)
                                                                    ⊢ LT.lt k (cs.leftInvSeq (CoxeterSystem.alternatingWord j i (HMul.hMul 2 p))). …
                                                                  -/
    MulAut.conj (s i) ((lis (alternatingWord j i (2 * p)))[k]'(by simp; omega)) := by
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
  rw [cs.getElem_leftInvSeq (alternatingWord i j (2 * p)) (k + 1) (by simp[h]),
    cs.getElem_leftInvSeq (alternatingWord j i (2 * p)) k (by simp[h]; omega)]
  simp only [MulAut.conj, listTake_succ_alternatingWord i j p k h, cs.wordProd_cons, mul_assoc,
    mul_inv_rev, inv_simple, MonoidHom.coe_mk, OneHom.coe_mk, MulEquiv.coe_mk, Equiv.coe_fn_mk,
    mul_right_inj, mul_left_inj]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i j : B
    p k : Nat
    h : LT.lt (HAdd.hAdd k 1) (HMul.hMul 2 p)
    ⊢ Eq (cs.simple (GetElem.getElem (CoxeterSystem.alternatingWord i j (HMul.hMul …
  -/
  rw [getElem_alternatingWord_swapIndices i j (2 * p) k]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i j : B
    p k : Nat
    h : LT.lt (HAdd.hAdd k 1) (HMul.hMul 2 p)
    ⊢ LT.lt (HAdd.hAdd k 1) (HMul.hMul 2 p)
  -/
  omega
  /-
    🎉 no goals
  -/


theorem getElem_leftInvSeq_alternatingWord
    (i j : B) (p k : ℕ) (h : k < 2 * p) :
                                               /-
                                                 B : Type u_1
                                                 W : Type u_2
                                                 inst✝ : Group W
                                                 M : CoxeterMatrix B
                                                 cs : CoxeterSystem M W
                                                 i j : B
                                                 p k : Nat
                                                 h : LT.lt k (HMul.hMul 2 p)
                                                 ⊢ LT.lt k (cs.leftInvSeq (CoxeterSystem.alternatingWord i j (HMul.hMul 2 p))). …
                                               -/
    (lis (alternatingWord i j (2 * p)))[k]'(by simp; omega) =
                                                     /-
                                                       🎉 no goals
                                                     -/
    π alternatingWord j i (2 * k + 1) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i j : B
    p k : Nat
    h : LT.lt k (HMul.hMul 2 p)
    ⊢ Eq (GetElem.getElem (cs.leftInvSeq (CoxeterSystem.alternatingWord i j (HMul. …
  -/
  revert i j
  induction k with
  | zero =>
    intro i j
    simp only [CoxeterSystem.getElem_leftInvSeq cs (alternatingWord i j (2 * p)) 0 (by simp [h]),
      take_zero, wordProd_nil, one_mul, inv_one, mul_one, alternatingWord, concat_eq_append,
      nil_append, wordProd_singleton]
    apply congr_arg
    simp only [getElem_alternatingWord i j (2 * p) 0 (by simp [h]), add_zero, even_two,
      Even.mul_right, ↓reduceIte]
  | succ k hk =>
    intro i j
    simp only [getElem_succ_leftInvSeq_alternatingWord cs i j p k h, hk (by omega),
      MulAut.conj_apply, inv_simple, alternatingWord_succ' j i, even_two, Even.mul_right,
      ↓reduceIte, wordProd_cons]
    rw [(by ring: 2 * (k + 1) = 2 * k + 1 + 1), alternatingWord_succ j i, wordProd_concat]
    simp [mul_assoc]


