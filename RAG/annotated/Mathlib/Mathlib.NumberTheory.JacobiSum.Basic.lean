/-- The *Jacobi sum* of two multiplicative characters on a finite commutative ring. -/
def jacobiSum (χ ψ : MulChar R R') : R' :=
  ∑ x : R, χ x * ψ (1 - x)


lemma jacobiSum_comm (χ ψ : MulChar R R') : jacobiSum χ ψ = jacobiSum ψ χ := by
  /-
    R : Type u_1
    R' : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    inst✝ : CommRing R'
    χ ψ : MulChar R R'
    ⊢ Eq (jacobiSum χ ψ) (jacobiSum ψ χ)
  -/
  simp only [jacobiSum, mul_comm (χ _)]
  /-
    R : Type u_1
    R' : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    inst✝ : CommRing R'
    χ ψ : MulChar R R'
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (ψ (HSub.hSub 1 x)) (χ x)) (Finset.un …
  -/
  rw [← (Equiv.subLeft 1).sum_comp]
  /-
    R : Type u_1
    R' : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    inst✝ : CommRing R'
    χ ψ : MulChar R R'
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (ψ (HSub.hSub 1 ((Equiv.subLeft 1) i) …
  -/
  simp only [Equiv.subLeft_apply, sub_sub_cancel]
  /-
    🎉 no goals
  -/


/-- The Jacobi sum is compatible with ring homomorphisms. -/
lemma jacobiSum_ringHomComp {R'' : Type*} [CommRing R''] (χ ψ : MulChar R R') (f : R' →+* R'') :
    jacobiSum (χ.ringHomComp f) (ψ.ringHomComp f) = f (jacobiSum χ ψ) := by
  simp only [jacobiSum, MulChar.ringHomComp, MulChar.coe_mk, MonoidHom.coe_mk, OneHom.coe_mk,
    map_sum, map_mul]


/-- The Jacobi sum of two multiplicative characters on a nontrivial finite commutative ring `F`
can be written as a sum over `F \ {0,1}`. -/
lemma jacobiSum_eq_sum_sdiff (χ ψ : MulChar F R) :
    jacobiSum χ ψ = ∑ x ∈ univ \ {0,1}, χ x * ψ (1 - x) := by
  simp only [jacobiSum, subset_univ, sum_sdiff_eq_sub, sub_eq_add_neg, self_eq_add_right,
    neg_eq_zero]
  /-
    F : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : Nontrivial F
    inst✝² : Fintype F
    inst✝¹ : DecidableEq F
    inst✝ : CommRing R
    χ ψ : MulChar F R
    ⊢ Eq ((Insert.insert 0 (Singleton.singleton 1)).sum fun x => HMul.hMul (χ x) ( …
  -/
  apply sum_eq_zero
  simp only [mem_insert, mem_singleton, forall_eq_or_imp, χ.map_zero, neg_zero, add_zero, map_one,
    mul_one, forall_eq, add_neg_cancel, ψ.map_zero, mul_zero, and_self]


private lemma jacobiSum_eq_aux (χ ψ : MulChar F R) :
    jacobiSum χ ψ = ∑ x : F, χ x + ∑ x : F, ψ x - Fintype.card F +
                      ∑ x ∈ univ \ {0, 1}, (χ x - 1) * (ψ (1 - x) - 1) := by
  /-
    F : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : Nontrivial F
    inst✝² : Fintype F
    inst✝¹ : DecidableEq F
    inst✝ : CommRing R
    χ ψ : MulChar F R
    ⊢ Eq (jacobiSum χ ψ) (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Finset.univ.sum fun x = …
  -/
  rw [jacobiSum]
  conv =>
    enter [1, 2, x]
    rw [show ∀ x y : R, x * y = x + y - 1 + (x - 1) * (y - 1) by intros; ring]
  /-
    F : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : Nontrivial F
    inst✝² : Fintype F
    inst✝¹ : DecidableEq F
    inst✝ : CommRing R
    χ ψ : MulChar F R
    ⊢ Eq (Finset.univ.sum fun x => HAdd.hAdd (HSub.hSub (HAdd.hAdd (χ x) (ψ (HSub. …
  -/
  rw [sum_add_distrib, sum_sub_distrib, sum_add_distrib]
  /-
    F : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : Nontrivial F
    inst✝² : Fintype F
    inst✝¹ : DecidableEq F
    inst✝ : CommRing R
    χ ψ : MulChar F R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Finset.univ.sum fun x => χ x) (Finset.u …
  -/
  conv => enter [1, 1, 1, 2, 2, x]; rw [← Equiv.subLeft_apply 1]
  rw [(Equiv.subLeft 1).sum_comp ψ, Fintype.card_eq_sum_ones, Nat.cast_sum, Nat.cast_one,
    sum_sdiff_eq_sub (subset_univ _), ← sub_zero (_ - _ + _), add_sub_assoc]
  /-
    F : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : Nontrivial F
    inst✝² : Fintype F
    inst✝¹ : DecidableEq F
    inst✝ : CommRing R
    χ ψ : MulChar F R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Finset.univ.sum fun x => χ x) (Finset.u …
  -/
  congr
  /-
    case e_a.e_a
    F : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : Nontrivial F
    inst✝² : Fintype F
    inst✝¹ : DecidableEq F
    inst✝ : CommRing R
    χ ψ : MulChar F R
    ⊢ Eq 0 ((Insert.insert 0 (Singleton.singleton 1)).sum fun x => HMul.hMul (HSub …
  -/
  rw [sum_pair zero_ne_one, sub_zero, ψ.map_one, χ.map_one, sub_self, mul_zero, zero_mul, add_zero]
  /-
    🎉 no goals
  -/


/-- The Jacobi sum of twice the trivial multiplicative character on a finite field `F`
equals `#F-2`. -/
theorem jacobiSum_trivial_trivial :
    jacobiSum (MulChar.trivial F R) (MulChar.trivial F R) = Fintype.card F - 2 := by
  classical
  rw [jacobiSum_eq_sum_sdiff]
  have : ∀ x ∈ univ \ {0, 1}, (MulChar.trivial F R) x * (MulChar.trivial F R) (1 - x) = 1 := by
    intros x hx
    rw [← map_mul, MulChar.trivial_apply, if_pos]
    simp only [mem_sdiff, mem_univ, mem_insert, mem_singleton, not_or, ← ne_eq, true_and] at hx
    simpa only [isUnit_iff_ne_zero, mul_ne_zero_iff, ne_eq, sub_eq_zero, @eq_comm _ _ x] using hx
  calc ∑ x ∈ univ \ {0, 1}, (MulChar.trivial F R) x * (MulChar.trivial F R) (1 - x)
  _ = ∑ _ ∈ univ \ {0, 1}, 1 := sum_congr rfl this
  _ = #(univ \ {0, 1}) := (cast_card _).symm
  _ = Fintype.card F - 2 := by
    rw [card_sdiff (subset_univ _), card_univ, card_pair zero_ne_one,
      Nat.cast_sub <| Nat.add_one_le_of_lt Fintype.one_lt_card, Nat.cast_two]


/-- If `1` is the trivial multiplicative character on a finite field `F`, then `J(1,1) = #F-2`. -/
theorem jacobiSum_one_one : jacobiSum (1 : MulChar F R) 1 = Fintype.card F - 2 :=
  jacobiSum_trivial_trivial


/-- If `χ` is a nontrivial multiplicative character on a finite field `F`, then `J(1,χ) = -1`. -/
theorem jacobiSum_one_nontrivial {χ : MulChar F R} (hχ : χ ≠ 1) : jacobiSum 1 χ = -1 := by
  classical
  have : ∑ x ∈ univ \ {0, 1}, ((1 : MulChar F R) x - 1) * (χ (1 - x) - 1) = 0 := by
    apply Finset.sum_eq_zero
    simp +contextual only [mem_sdiff, mem_univ, mem_insert, mem_singleton,
      not_or, ← isUnit_iff_ne_zero, true_and, MulChar.one_apply, sub_self, zero_mul, and_imp,
      implies_true]
  simp only [jacobiSum_eq_aux, MulChar.sum_one_eq_card_units, MulChar.sum_eq_zero_of_ne_one hχ,
    add_zero, Fintype.card_eq_card_units_add_one (α := F), Nat.cast_add, Nat.cast_one,
    sub_add_cancel_left, this]


/-- If `χ` is a nontrivial multiplicative character on a finite field `F`,
then `J(χ,χ⁻¹) = -χ(-1)`. -/
theorem jacobiSum_nontrivial_inv {χ : MulChar F R} (hχ : χ ≠ 1) : jacobiSum χ χ⁻¹ = -χ (-1) := by
  classical
  rw [jacobiSum]
  conv => enter [1, 2, x]; rw [MulChar.inv_apply', ← map_mul, ← div_eq_mul_inv]
  rw [sum_eq_sum_diff_singleton_add (mem_univ (1 : F)), sub_self, div_zero, χ.map_zero, add_zero]
  have : ∑ x ∈ univ \ {1}, χ (x / (1 - x)) = ∑ x ∈ univ \ {-1}, χ x := by
    refine sum_bij' (fun a _ ↦ a / (1 - a)) (fun b _ ↦ b / (1 + b)) (fun x hx ↦ ?_)
      (fun y hy ↦ ?_) (fun x hx ↦ ?_) (fun y hy ↦ ?_) (fun _ _ ↦ rfl)
    · simp only [mem_sdiff, mem_univ, mem_singleton, true_and] at hx ⊢
      rw [div_eq_iff <| sub_ne_zero.mpr ((ne_eq ..).symm ▸ hx).symm, mul_sub, mul_one,
        neg_one_mul, sub_neg_eq_add, self_eq_add_left, neg_eq_zero]
      exact one_ne_zero
    · simp only [mem_sdiff, mem_univ, mem_singleton, true_and] at hy ⊢
      rw [div_eq_iff fun h ↦ hy <| eq_neg_of_add_eq_zero_right h, one_mul, self_eq_add_left]
      exact one_ne_zero
    · simp only [mem_sdiff, mem_univ, mem_singleton, true_and] at hx
      rw [eq_comm, ← sub_eq_zero] at hx
      field_simp
    · simp only [mem_sdiff, mem_univ, mem_singleton, true_and] at hy
      rw [eq_comm, neg_eq_iff_eq_neg, ← sub_eq_zero, sub_neg_eq_add] at hy
      field_simp
  rw [this, ← add_eq_zero_iff_eq_neg, ← sum_eq_sum_diff_singleton_add (mem_univ (-1 : F))]
  exact MulChar.sum_eq_zero_of_ne_one hχ


/-- If `χ` and `φ` are multiplicative characters on a finite field `F` such that
`χφ` is nontrivial, then `g(χφ) * J(χ,φ) = g(χ) * g(φ)`. -/
theorem jacobiSum_mul_nontrivial {χ φ : MulChar F R} (h : χ * φ ≠ 1) (ψ : AddChar F R) :
    gaussSum (χ * φ) ψ * jacobiSum χ φ = gaussSum χ ψ * gaussSum φ ψ := by
  classical
  rw [gaussSum_mul _ _ ψ, sum_eq_sum_diff_singleton_add (mem_univ (0 : F))]
  conv =>
    enter [2, 2, 2, x]
    rw [zero_sub, neg_eq_neg_one_mul x, map_mul, mul_left_comm (χ x) (φ (-1)),
      ← MulChar.mul_apply, ψ.map_zero_eq_one, mul_one]
  rw [← mul_sum _ _ (φ (-1)), MulChar.sum_eq_zero_of_ne_one h, mul_zero, add_zero]
  have sum_eq : ∀ t ∈ univ \ {0}, (∑ x : F, χ x * φ (t - x)) * ψ t =
      (∑ y : F, χ (t * y) * φ (t - (t * y))) * ψ t := by
    intro t ht
    simp only [mem_sdiff, mem_univ, mem_singleton, true_and] at ht
    exact congrArg (· * ψ t) (Equiv.sum_comp (Equiv.mulLeft₀ t ht) _).symm
  simp_rw [← sum_mul, sum_congr rfl sum_eq, ← mul_one_sub, map_mul, mul_assoc]
  conv => enter [2, 2, t, 1, 2, x, 2]; rw [← mul_assoc, mul_comm (χ x) (φ t)]
  simp_rw [← mul_assoc, ← MulChar.mul_apply, mul_assoc, ← mul_sum, mul_right_comm]
  rw [← jacobiSum, ← sum_mul, gaussSum, sum_eq_sum_diff_singleton_add (mem_univ (0 : F)),
    (χ * φ).map_zero, zero_mul, add_zero]


/-- If `χ` and `φ` are multiplicative characters on a finite field `F` with values
in another field `F'` and such that `χφ` is nontrivial, then `J(χ,φ) = g(χ) * g(φ) / g(χφ)`. -/
theorem jacobiSum_eq_gaussSum_mul_gaussSum_div_gaussSum (h : (Fintype.card F : F') ≠ 0)
    {χ φ : MulChar F F'} (hχφ : χ * φ ≠ 1) {ψ : AddChar F F'} (hψ : ψ.IsPrimitive) :
    jacobiSum χ φ = gaussSum χ ψ * gaussSum φ ψ / gaussSum (χ * φ) ψ := by
  /-
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (↑(Fintype.card F)) 0
    χ φ : MulChar F F'
    hχφ : Ne (HMul.hMul χ φ) 1
    ψ : AddChar F F'
    hψ : ψ.IsPrimitive
    ⊢ Eq (jacobiSum χ φ) (HDiv.hDiv (HMul.hMul (gaussSum χ ψ) (gaussSum φ ψ)) (gau …
  -/
  rw [eq_div_iff <| gaussSum_ne_zero_of_nontrivial h hχφ hψ, mul_comm]
  /-
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (↑(Fintype.card F)) 0
    χ φ : MulChar F F'
    hχφ : Ne (HMul.hMul χ φ) 1
    ψ : AddChar F F'
    hψ : ψ.IsPrimitive
    ⊢ Eq (HMul.hMul (gaussSum (HMul.hMul χ φ) ψ) (jacobiSum χ φ)) (HMul.hMul (gaus …
  -/
  exact jacobiSum_mul_nontrivial hχφ ψ
  /-
    🎉 no goals
  -/


open AddChar MulChar in
/-- If `χ` and `φ` are multiplicative characters on a finite field `F` with values in another
field `F'` such that `χ`, `φ` and `χφ` are all nontrivial and `char F' ≠ char F`, then
`J(χ,φ) * J(χ⁻¹,φ⁻¹) = #F` (in `F'`). -/
lemma jacobiSum_mul_jacobiSum_inv (h : ringChar F' ≠ ringChar F) {χ φ : MulChar F F'} (hχ : χ ≠ 1)
    (hφ : φ ≠ 1) (hχφ : χ * φ ≠ 1) :
    jacobiSum χ φ * jacobiSum χ⁻¹ φ⁻¹ = Fintype.card F := by
  /-
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    ⊢ Eq (HMul.hMul (jacobiSum χ φ) (jacobiSum (Inv.inv χ) (Inv.inv φ))) ↑(Fintype …
  -/
  obtain ⟨n, hp, hc⟩ := FiniteField.card F (ringChar F)
  /-
    case intro.intro
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ⊢ Eq (HMul.hMul (jacobiSum χ φ) (jacobiSum (Inv.inv χ) (Inv.inv φ))) ↑(Fintype …
  -/
  let ψ := FiniteField.primitiveChar F F' h   -- obtain primitive additive character `ψ : F → FF'`
  /-
    case intro.intro
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' h
    ⊢ Eq (HMul.hMul (jacobiSum χ φ) (jacobiSum (Inv.inv χ) (Inv.inv φ))) ↑(Fintype …
  -/
  let FF' := CyclotomicField ψ.n F'           -- the target field of `ψ`
  /-
    case intro.intro
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' h
    FF' : Type u_2 := CyclotomicField ψ.n F'
    ⊢ Eq (HMul.hMul (jacobiSum χ φ) (jacobiSum (Inv.inv χ) (Inv.inv φ))) ↑(Fintype …
  -/
  let χ' := χ.ringHomComp (algebraMap F' FF') -- consider `χ` and `φ` as characters `F → FF'`
  /-
    case intro.intro
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' h
    FF' : Type u_2 := CyclotomicField ψ.n F'
    χ' : MulChar F FF' := χ.ringHomComp (algebraMap F' FF')
    ⊢ Eq (HMul.hMul (jacobiSum χ φ) (jacobiSum (Inv.inv χ) (Inv.inv φ))) ↑(Fintype …
  -/
  let φ' := φ.ringHomComp (algebraMap F' FF')
  /-
    case intro.intro
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' h
    FF' : Type u_2 := CyclotomicField ψ.n F'
    χ' : MulChar F FF' := χ.ringHomComp (algebraMap F' FF')
    φ' : MulChar F FF' := φ.ringHomComp (algebraMap F' FF')
    ⊢ Eq (HMul.hMul (jacobiSum χ φ) (jacobiSum (Inv.inv χ) (Inv.inv φ))) ↑(Fintype …
  -/
  have hinj := (algebraMap F' FF').injective
  /-
    case intro.intro
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' h
    FF' : Type u_2 := CyclotomicField ψ.n F'
    χ' : MulChar F FF' := χ.ringHomComp (algebraMap F' FF')
    φ' : MulChar F FF' := φ.ringHomComp (algebraMap F' FF')
    hinj : Function.Injective ⇑(algebraMap F' FF')
    ⊢ Eq (HMul.hMul (jacobiSum χ φ) (jacobiSum (Inv.inv χ) (Inv.inv φ))) ↑(Fintype …
  -/
  apply hinj
  /-
    case intro.intro.a
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' h
    FF' : Type u_2 := CyclotomicField ψ.n F'
    χ' : MulChar F FF' := χ.ringHomComp (algebraMap F' FF')
    φ' : MulChar F FF' := φ.ringHomComp (algebraMap F' FF')
    hinj : Function.Injective ⇑(algebraMap F' FF')
    ⊢ Eq ((algebraMap F' FF') (HMul.hMul (jacobiSum χ φ) (jacobiSum (Inv.inv χ) (I …
  -/
  rw [map_mul, ← jacobiSum_ringHomComp, ← jacobiSum_ringHomComp]
  have Hχφ : χ' * φ' ≠ 1 := by
    rw [← ringHomComp_mul]
    exact (MulChar.ringHomComp_ne_one_iff hinj).mpr hχφ
  have Hχφ' : χ'⁻¹ * φ'⁻¹ ≠ 1 := by
    rwa [← mul_inv, inv_ne_one]
  /-
    case intro.intro.a
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' h
    FF' : Type u_2 := CyclotomicField ψ.n F'
    χ' : MulChar F FF' := χ.ringHomComp (algebraMap F' FF')
    φ' : MulChar F FF' := φ.ringHomComp (algebraMap F' FF')
    hinj : Function.Injective ⇑(algebraMap F' FF')
    Hχφ : Ne (HMul.hMul χ' φ') 1
    Hχφ' : Ne (HMul.hMul (Inv.inv χ') (Inv.inv φ')) 1
    ⊢ Eq (HMul.hMul (jacobiSum (χ.ringHomComp (algebraMap F' FF')) (φ.ringHomComp  …
  -/
  have Hχ : χ' ≠ 1 := (MulChar.ringHomComp_ne_one_iff hinj).mpr hχ
  /-
    case intro.intro.a
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' h
    FF' : Type u_2 := CyclotomicField ψ.n F'
    χ' : MulChar F FF' := χ.ringHomComp (algebraMap F' FF')
    φ' : MulChar F FF' := φ.ringHomComp (algebraMap F' FF')
    hinj : Function.Injective ⇑(algebraMap F' FF')
    Hχφ : Ne (HMul.hMul χ' φ') 1
    Hχφ' : Ne (HMul.hMul (Inv.inv χ') (Inv.inv φ')) 1
    Hχ : Ne χ' 1
    ⊢ Eq (HMul.hMul (jacobiSum (χ.ringHomComp (algebraMap F' FF')) (φ.ringHomComp  …
  -/
  have Hφ : φ' ≠ 1 := (MulChar.ringHomComp_ne_one_iff hinj).mpr hφ
  have Hcard : (Fintype.card F : FF') ≠ 0 := by
    intro H
    simp only [hc, Nat.cast_pow, ne_eq, PNat.ne_zero, not_false_eq_true, pow_eq_zero_iff] at H
    exact h <| (Algebra.ringChar_eq F' FF').trans <| CharP.ringChar_of_prime_eq_zero hp H
  /-
    case intro.intro.a
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' h
    FF' : Type u_2 := CyclotomicField ψ.n F'
    χ' : MulChar F FF' := χ.ringHomComp (algebraMap F' FF')
    φ' : MulChar F FF' := φ.ringHomComp (algebraMap F' FF')
    hinj : Function.Injective ⇑(algebraMap F' FF')
    Hχφ : Ne (HMul.hMul χ' φ') 1
    Hχφ' : Ne (HMul.hMul (Inv.inv χ') (Inv.inv φ')) 1
    Hχ : Ne χ' 1
    Hφ : Ne φ' 1
    Hcard : Ne (↑(Fintype.card F)) 0
    ⊢ Eq (HMul.hMul (jacobiSum (χ.ringHomComp (algebraMap F' FF')) (φ.ringHomComp  …
  -/
  have H := (gaussSum_mul_gaussSum_eq_card Hχφ ψ.prim).trans_ne Hcard
  apply_fun (gaussSum (χ' * φ') ψ.char * gaussSum (χ' * φ')⁻¹ ψ.char⁻¹ * ·)
    using mul_right_injective₀ H
  /-
    case intro.intro.a
    F : Type u_1
    F' : Type u_2
    inst✝² : Fintype F
    inst✝¹ : Field F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    χ φ : MulChar F F'
    hχ : Ne χ 1
    hφ : Ne φ 1
    hχφ : Ne (HMul.hMul χ φ) 1
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' h
    FF' : Type u_2 := CyclotomicField ψ.n F'
    χ' : MulChar F FF' := χ.ringHomComp (algebraMap F' FF')
    φ' : MulChar F FF' := φ.ringHomComp (algebraMap F' FF')
    hinj : Function.Injective ⇑(algebraMap F' FF')
    Hχφ : Ne (HMul.hMul χ' φ') 1
    Hχφ' : Ne (HMul.hMul (Inv.inv χ') (Inv.inv φ')) 1
    Hχ : Ne χ' 1
    Hφ : Ne φ' 1
    Hcard : Ne (↑(Fintype.card F)) 0
    H : Ne (HMul.hMul (gaussSum (HMul.hMul χ' φ') ψ.char) (gaussSum (Inv.inv (HMul …
    ⊢ Eq ((fun x => HMul.hMul (HMul.hMul (gaussSum (HMul.hMul χ' φ') ψ.char) (gaus …
  -/
  simp only
  rw [mul_mul_mul_comm, jacobiSum_mul_nontrivial Hχφ, mul_inv, ← ringHomComp_inv,
    ← ringHomComp_inv, jacobiSum_mul_nontrivial Hχφ', map_natCast, ← mul_mul_mul_comm,
    gaussSum_mul_gaussSum_eq_card Hχ ψ.prim, gaussSum_mul_gaussSum_eq_card Hφ ψ.prim,
    ← mul_inv, gaussSum_mul_gaussSum_eq_card Hχφ ψ.prim]


/-- If `χ` and `φ` are multiplicative characters on a finite field `F` satisfying `χ^n = φ^n = 1`
and with values in an integral domain `R`, and `μ` is a primitive `n`th root of unity in `R`,
then the Jacobi sum `J(χ,φ)` is in `ℤ[μ] ⊆ R`. -/
lemma jacobiSum_mem_algebraAdjoin_of_pow_eq_one {n : ℕ} [NeZero n] {χ φ : MulChar F R}
    (hχ : χ ^ n = 1) (hφ : φ ^ n = 1) {μ : R} (hμ : IsPrimitiveRoot μ n) :
    jacobiSum χ φ ∈ Algebra.adjoin ℤ {μ} :=
  Subalgebra.sum_mem _ fun _ _ ↦ Subalgebra.mul_mem _
    (MulChar.apply_mem_algebraAdjoin_of_pow_eq_one hχ hμ _)
    (MulChar.apply_mem_algebraAdjoin_of_pow_eq_one hφ hμ _)


open Algebra in
private
lemma MulChar.exists_apply_sub_one_eq_mul_sub_one {n : ℕ} [NeZero n] {χ : MulChar F R} {μ : R}
    (hχ : χ ^ n = 1) (hμ : IsPrimitiveRoot μ n) {x : F} (hx : x ≠ 0) :
    ∃ z ∈ Algebra.adjoin ℤ {μ}, χ x - 1 = z * (μ - 1) := by
  /-
    F : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype F
    inst✝³ : Field F
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    χ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hμ : IsPrimitiveRoot μ n
    x : F
    hx : Ne x 0
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  obtain ⟨k, _, hk⟩ := exists_apply_eq_pow hχ hμ hx
  /-
    case intro.intro
    F : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype F
    inst✝³ : Field F
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    χ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hμ : IsPrimitiveRoot μ n
    x : F
    hx : Ne x 0
    k : Nat
    left✝ : LT.lt k n
    hk : Eq (χ x) (HPow.hPow μ k)
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  refine hk ▸ ⟨(Finset.range k).sum (μ ^ ·), ?_, (geom_sum_mul μ k).symm⟩
  /-
    case intro.intro
    F : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype F
    inst✝³ : Field F
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    χ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hμ : IsPrimitiveRoot μ n
    x : F
    hx : Ne x 0
    k : Nat
    left✝ : LT.lt k n
    hk : Eq (χ x) (HPow.hPow μ k)
    ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) ((Finset.range k …
  -/
  exact Subalgebra.sum_mem _ fun m _ ↦ Subalgebra.pow_mem _ (self_mem_adjoin_singleton _ μ) _
  /-
    🎉 no goals
  -/


private
lemma MulChar.exists_apply_sub_one_mul_apply_sub_one {n : ℕ} [NeZero n] {χ ψ : MulChar F R}
    {μ : R} (hχ : χ ^ n = 1) (hψ : ψ ^ n = 1) (hμ : IsPrimitiveRoot μ n) (x : F) :
    ∃ z ∈ Algebra.adjoin ℤ {μ}, (χ x - 1) * (ψ (1 - x) - 1) = z * (μ - 1) ^ 2 := by
  /-
    F : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype F
    inst✝³ : Field F
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    χ ψ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hψ : Eq (HPow.hPow ψ n) 1
    hμ : IsPrimitiveRoot μ n
    x : F
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  rcases eq_or_ne x 0 with rfl | hx₀
    /-
      case inl
      F : Type u_1
      R : Type u_2
      inst✝⁴ : Fintype F
      inst✝³ : Field F
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      χ ψ : MulChar F R
      μ : R
      hχ : Eq (HPow.hPow χ n) 1
      hψ : Eq (HPow.hPow ψ n) 1
      hμ : IsPrimitiveRoot μ n
      ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
    -/
  · exact ⟨0, Subalgebra.zero_mem _, by rw [sub_zero, ψ.map_one, sub_self, mul_zero, zero_mul]⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    F : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype F
    inst✝³ : Field F
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    χ ψ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hψ : Eq (HPow.hPow ψ n) 1
    hμ : IsPrimitiveRoot μ n
    x : F
    hx₀ : Ne x 0
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  rcases eq_or_ne x 1 with rfl | hx₁
    /-
      case inr.inl
      F : Type u_1
      R : Type u_2
      inst✝⁴ : Fintype F
      inst✝³ : Field F
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      χ ψ : MulChar F R
      μ : R
      hχ : Eq (HPow.hPow χ n) 1
      hψ : Eq (HPow.hPow ψ n) 1
      hμ : IsPrimitiveRoot μ n
      hx₀ : Ne 1 0
      ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
    -/
  · exact ⟨0, Subalgebra.zero_mem _, by rw [χ.map_one, sub_self, zero_mul, zero_mul]⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    F : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype F
    inst✝³ : Field F
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    χ ψ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hψ : Eq (HPow.hPow ψ n) 1
    hμ : IsPrimitiveRoot μ n
    x : F
    hx₀ : Ne x 0
    hx₁ : Ne x 1
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  obtain ⟨z₁, hz₁, Hz₁⟩ := MulChar.exists_apply_sub_one_eq_mul_sub_one hχ hμ hx₀
  obtain ⟨z₂, hz₂, Hz₂⟩ :=
    MulChar.exists_apply_sub_one_eq_mul_sub_one hψ hμ (sub_ne_zero_of_ne hx₁.symm)
  /-
    case inr.inr.intro.intro.intro.intro
    F : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype F
    inst✝³ : Field F
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    χ ψ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hψ : Eq (HPow.hPow ψ n) 1
    hμ : IsPrimitiveRoot μ n
    x : F
    hx₀ : Ne x 0
    hx₁ : Ne x 1
    z₁ : R
    hz₁ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₁
    Hz₁ : Eq (HSub.hSub (χ x) 1) (HMul.hMul z₁ (HSub.hSub μ 1))
    z₂ : R
    hz₂ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₂
    Hz₂ : Eq (HSub.hSub (ψ (HSub.hSub 1 x)) 1) (HMul.hMul z₂ (HSub.hSub μ 1))
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  rewrite [Hz₁, Hz₂, sq]
  /-
    case inr.inr.intro.intro.intro.intro
    F : Type u_1
    R : Type u_2
    inst✝⁴ : Fintype F
    inst✝³ : Field F
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    χ ψ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hψ : Eq (HPow.hPow ψ n) 1
    hμ : IsPrimitiveRoot μ n
    x : F
    hx₀ : Ne x 0
    hx₁ : Ne x 1
    z₁ : R
    hz₁ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₁
    Hz₁ : Eq (HSub.hSub (χ x) 1) (HMul.hMul z₁ (HSub.hSub μ 1))
    z₂ : R
    hz₂ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₂
    Hz₂ : Eq (HSub.hSub (ψ (HSub.hSub 1 x)) 1) (HMul.hMul z₂ (HSub.hSub μ 1))
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  exact ⟨z₁ * z₂, Subalgebra.mul_mem _ hz₁ hz₂, mul_mul_mul_comm ..⟩
  /-
    🎉 no goals
  -/


/-- If `χ` and `ψ` are multiplicative characters of order dividing `n` on a finite field `F`
with values in an integral domain `R` and `μ` is a primitive `n`th root of unity in `R`,
then `J(χ,ψ) = -1 + z*(μ - 1)^2` for some `z ∈ ℤ[μ] ⊆ R`. (We assume that `#F ≡ 1 mod n`.)
Note that we do not state this as a divisibility in `R`, as this would give a weaker statement. -/
lemma exists_jacobiSum_eq_neg_one_add {n : ℕ} (hn : 2 < n) {χ ψ : MulChar F R}
    {μ : R} (hχ : χ ^ n = 1) (hψ : ψ ^ n = 1) (hn' : n ∣ Fintype.card F - 1)
    (hμ : IsPrimitiveRoot μ n) :
    ∃ z ∈ Algebra.adjoin ℤ {μ}, jacobiSum χ ψ = -1 + z * (μ - 1) ^ 2 := by
  /-
    F : Type u_1
    R : Type u_2
    inst✝³ : Fintype F
    inst✝² : Field F
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    hn : LT.lt 2 n
    χ ψ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hψ : Eq (HPow.hPow ψ n) 1
    hn' : Dvd.dvd n (HSub.hSub (Fintype.card F) 1)
    hμ : IsPrimitiveRoot μ n
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  obtain ⟨q, hq⟩ := hn'
  /-
    case intro
    F : Type u_1
    R : Type u_2
    inst✝³ : Fintype F
    inst✝² : Field F
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    hn : LT.lt 2 n
    χ ψ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hψ : Eq (HPow.hPow ψ n) 1
    hμ : IsPrimitiveRoot μ n
    q : Nat
    hq : Eq (HSub.hSub (Fintype.card F) 1) (HMul.hMul n q)
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  rw [Nat.sub_eq_iff_eq_add NeZero.one_le] at hq
  /-
    case intro
    F : Type u_1
    R : Type u_2
    inst✝³ : Fintype F
    inst✝² : Field F
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    hn : LT.lt 2 n
    χ ψ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hψ : Eq (HPow.hPow ψ n) 1
    hμ : IsPrimitiveRoot μ n
    q : Nat
    hq : Eq (Fintype.card F) (HAdd.hAdd (HMul.hMul n q) 1)
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  obtain ⟨z₁, hz₁, Hz₁⟩ := hμ.self_sub_one_pow_dvd_order hn
  /-
    case intro.intro.intro
    F : Type u_1
    R : Type u_2
    inst✝³ : Fintype F
    inst✝² : Field F
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    hn : LT.lt 2 n
    χ ψ : MulChar F R
    μ : R
    hχ : Eq (HPow.hPow χ n) 1
    hψ : Eq (HPow.hPow ψ n) 1
    hμ : IsPrimitiveRoot μ n
    q : Nat
    hq : Eq (Fintype.card F) (HAdd.hAdd (HMul.hMul n q) 1)
    z₁ : R
    hz₁ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₁
    Hz₁ : Eq (↑n) (HMul.hMul z₁ (HPow.hPow (HSub.hSub μ 1) 2))
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  by_cases hχ₀ : χ = 1 <;> by_cases hψ₀ : ψ = 1
    /-
      case pos
      F : Type u_1
      R : Type u_2
      inst✝³ : Fintype F
      inst✝² : Field F
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      hn : LT.lt 2 n
      χ ψ : MulChar F R
      μ : R
      hχ : Eq (HPow.hPow χ n) 1
      hψ : Eq (HPow.hPow ψ n) 1
      hμ : IsPrimitiveRoot μ n
      q : Nat
      hq : Eq (Fintype.card F) (HAdd.hAdd (HMul.hMul n q) 1)
      z₁ : R
      hz₁ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₁
      Hz₁ : Eq (↑n) (HMul.hMul z₁ (HPow.hPow (HSub.hSub μ 1) 2))
      hχ₀ : Eq χ 1
      hψ₀ : Eq ψ 1
      ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
    -/
  · rw [hχ₀, hψ₀, jacobiSum_one_one]
    /-
      case pos
      F : Type u_1
      R : Type u_2
      inst✝³ : Fintype F
      inst✝² : Field F
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      hn : LT.lt 2 n
      χ ψ : MulChar F R
      μ : R
      hχ : Eq (HPow.hPow χ n) 1
      hψ : Eq (HPow.hPow ψ n) 1
      hμ : IsPrimitiveRoot μ n
      q : Nat
      hq : Eq (Fintype.card F) (HAdd.hAdd (HMul.hMul n q) 1)
      z₁ : R
      hz₁ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₁
      Hz₁ : Eq (↑n) (HMul.hMul z₁ (HPow.hPow (HSub.hSub μ 1) 2))
      hχ₀ : Eq χ 1
      hψ₀ : Eq ψ 1
      ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
    -/
    refine ⟨q * z₁, Subalgebra.mul_mem _ (Subalgebra.natCast_mem _ q) hz₁, ?_⟩
    /-
      case pos
      F : Type u_1
      R : Type u_2
      inst✝³ : Fintype F
      inst✝² : Field F
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      hn : LT.lt 2 n
      χ ψ : MulChar F R
      μ : R
      hχ : Eq (HPow.hPow χ n) 1
      hψ : Eq (HPow.hPow ψ n) 1
      hμ : IsPrimitiveRoot μ n
      q : Nat
      hq : Eq (Fintype.card F) (HAdd.hAdd (HMul.hMul n q) 1)
      z₁ : R
      hz₁ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₁
      Hz₁ : Eq (↑n) (HMul.hMul z₁ (HPow.hPow (HSub.hSub μ 1) 2))
      hχ₀ : Eq χ 1
      hψ₀ : Eq ψ 1
      ⊢ Eq (HSub.hSub (↑(Fintype.card F)) 2) (HAdd.hAdd (-1) (HMul.hMul (HMul.hMul ( …
    -/
    rw [hq, Nat.cast_add, Nat.cast_mul, Hz₁]
    /-
      case pos
      F : Type u_1
      R : Type u_2
      inst✝³ : Fintype F
      inst✝² : Field F
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      hn : LT.lt 2 n
      χ ψ : MulChar F R
      μ : R
      hχ : Eq (HPow.hPow χ n) 1
      hψ : Eq (HPow.hPow ψ n) 1
      hμ : IsPrimitiveRoot μ n
      q : Nat
      hq : Eq (Fintype.card F) (HAdd.hAdd (HMul.hMul n q) 1)
      z₁ : R
      hz₁ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₁
      Hz₁ : Eq (↑n) (HMul.hMul z₁ (HPow.hPow (HSub.hSub μ 1) 2))
      hχ₀ : Eq χ 1
      hψ₀ : Eq ψ 1
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z₁ (HPow.hPow (HSub.hSub μ 1) …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type u_1
      R : Type u_2
      inst✝³ : Fintype F
      inst✝² : Field F
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      hn : LT.lt 2 n
      χ ψ : MulChar F R
      μ : R
      hχ : Eq (HPow.hPow χ n) 1
      hψ : Eq (HPow.hPow ψ n) 1
      hμ : IsPrimitiveRoot μ n
      q : Nat
      hq : Eq (Fintype.card F) (HAdd.hAdd (HMul.hMul n q) 1)
      z₁ : R
      hz₁ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₁
      Hz₁ : Eq (↑n) (HMul.hMul z₁ (HPow.hPow (HSub.hSub μ 1) 2))
      hχ₀ : Eq χ 1
      hψ₀ : Not (Eq ψ 1)
      ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
    -/
  · refine ⟨0, Subalgebra.zero_mem _, ?_⟩
    /-
      case neg
      F : Type u_1
      R : Type u_2
      inst✝³ : Fintype F
      inst✝² : Field F
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      hn : LT.lt 2 n
      χ ψ : MulChar F R
      μ : R
      hχ : Eq (HPow.hPow χ n) 1
      hψ : Eq (HPow.hPow ψ n) 1
      hμ : IsPrimitiveRoot μ n
      q : Nat
      hq : Eq (Fintype.card F) (HAdd.hAdd (HMul.hMul n q) 1)
      z₁ : R
      hz₁ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₁
      Hz₁ : Eq (↑n) (HMul.hMul z₁ (HPow.hPow (HSub.hSub μ 1) 2))
      hχ₀ : Eq χ 1
      hψ₀ : Not (Eq ψ 1)
      ⊢ Eq (jacobiSum χ ψ) (HAdd.hAdd (-1) (HMul.hMul 0 (HPow.hPow (HSub.hSub μ 1) 2 …
    -/
    rw [hχ₀, jacobiSum_one_nontrivial hψ₀, zero_mul, add_zero]
    /-
      🎉 no goals
    -/
    /-
      case pos
      F : Type u_1
      R : Type u_2
      inst✝³ : Fintype F
      inst✝² : Field F
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      hn : LT.lt 2 n
      χ ψ : MulChar F R
      μ : R
      hχ : Eq (HPow.hPow χ n) 1
      hψ : Eq (HPow.hPow ψ n) 1
      hμ : IsPrimitiveRoot μ n
      q : Nat
      hq : Eq (Fintype.card F) (HAdd.hAdd (HMul.hMul n q) 1)
      z₁ : R
      hz₁ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₁
      Hz₁ : Eq (↑n) (HMul.hMul z₁ (HPow.hPow (HSub.hSub μ 1) 2))
      hχ₀ : Not (Eq χ 1)
      hψ₀ : Eq ψ 1
      ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
    -/
  · refine ⟨0, Subalgebra.zero_mem _, ?_⟩
    /-
      case pos
      F : Type u_1
      R : Type u_2
      inst✝³ : Fintype F
      inst✝² : Field F
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      hn : LT.lt 2 n
      χ ψ : MulChar F R
      μ : R
      hχ : Eq (HPow.hPow χ n) 1
      hψ : Eq (HPow.hPow ψ n) 1
      hμ : IsPrimitiveRoot μ n
      q : Nat
      hq : Eq (Fintype.card F) (HAdd.hAdd (HMul.hMul n q) 1)
      z₁ : R
      hz₁ : Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) z₁
      Hz₁ : Eq (↑n) (HMul.hMul z₁ (HPow.hPow (HSub.hSub μ 1) 2))
      hχ₀ : Not (Eq χ 1)
      hψ₀ : Eq ψ 1
      ⊢ Eq (jacobiSum χ ψ) (HAdd.hAdd (-1) (HMul.hMul 0 (HPow.hPow (HSub.hSub μ 1) 2 …
    -/
    rw [jacobiSum_comm, hψ₀, jacobiSum_one_nontrivial hχ₀, zero_mul, add_zero]
    /-
      🎉 no goals
    -/
  · classical
    rw [jacobiSum_eq_aux, MulChar.sum_eq_zero_of_ne_one hχ₀, MulChar.sum_eq_zero_of_ne_one hψ₀, hq]
    have : NeZero n := ⟨by omega⟩
    have H := MulChar.exists_apply_sub_one_mul_apply_sub_one hχ hψ hμ
    have Hcs x := (H x).choose_spec
    refine ⟨-q * z₁ + ∑ x ∈ (univ \ {0, 1} : Finset F), (H x).choose, ?_, ?_⟩
    · refine Subalgebra.add_mem _ (Subalgebra.mul_mem _ (Subalgebra.neg_mem _ ?_) hz₁) ?_
      · exact Subalgebra.natCast_mem ..
      · exact Subalgebra.sum_mem _ fun x _ ↦ (Hcs x).1
    · conv => enter [1, 2, 2, x]; rw [(Hcs x).2]
      rw [← Finset.sum_mul, Nat.cast_add, Nat.cast_mul, Hz₁]
      ring


lemma gaussSum_pow_eq_prod_jacobiSum_aux (χ : MulChar F R) (ψ : AddChar F R) {n : ℕ}
    (hn₁ : 0 < n) (hn₂ : n < orderOf χ) :
    gaussSum χ ψ ^ n = gaussSum (χ ^ n) ψ * ∏ j ∈ Ico 1 n, jacobiSum χ (χ ^ j) := by
  induction n, hn₁ using Nat.le_induction with
  | base => simp only [pow_one, le_refl, Ico_eq_empty_of_le, prod_empty, mul_one]
  | succ n hn ih =>
      specialize ih <| lt_trans (Nat.lt_succ_self n) hn₂
      have gauss_rw : gaussSum (χ ^ n) ψ * gaussSum χ ψ =
            jacobiSum χ (χ ^ n) * gaussSum (χ ^ (n + 1)) ψ := by
        have hχn : χ * (χ ^ n) ≠ 1 :=
          pow_succ' χ n ▸ pow_ne_one_of_lt_orderOf n.add_one_ne_zero hn₂
        rw [mul_comm, ← jacobiSum_mul_nontrivial hχn, mul_comm, ← pow_succ']
      apply_fun (· * gaussSum χ ψ) at ih
      rw [mul_right_comm, ← pow_succ, gauss_rw] at ih
      rw [ih, Finset.prod_Ico_succ_top hn, mul_rotate, mul_assoc]


/-- If `χ` is a multiplicative character of order `n ≥ 2` on a finite field `F`,
then `g(χ)^n = χ(-1) * #F * J(χ,χ) * J(χ,χ²) * ... * J(χ,χⁿ⁻²)`. -/
theorem gaussSum_pow_eq_prod_jacobiSum {χ : MulChar F R} {ψ : AddChar F R} (hχ : 2 ≤ orderOf χ)
    (hψ : ψ.IsPrimitive) :
    gaussSum χ ψ ^ orderOf χ =
      χ (-1) * Fintype.card F * ∏ i ∈ Ico 1 (orderOf χ - 1), jacobiSum χ (χ ^ i) := by
  /-
    F : Type u_1
    R : Type u_2
    inst✝³ : Fintype F
    inst✝² : Field F
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    χ : MulChar F R
    ψ : AddChar F R
    hχ : LE.le 2 (orderOf χ)
    hψ : ψ.IsPrimitive
    ⊢ Eq (HPow.hPow (gaussSum χ ψ) (orderOf χ)) (HMul.hMul (HMul.hMul (χ (-1)) ↑(F …
  -/
  have := gaussSum_pow_eq_prod_jacobiSum_aux χ ψ (n := orderOf χ - 1) (by omega) (by omega)
  /-
    F : Type u_1
    R : Type u_2
    inst✝³ : Fintype F
    inst✝² : Field F
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    χ : MulChar F R
    ψ : AddChar F R
    hχ : LE.le 2 (orderOf χ)
    hψ : ψ.IsPrimitive
    this : Eq (HPow.hPow (gaussSum χ ψ) (HSub.hSub (orderOf χ) 1)) (HMul.hMul (gau …
    ⊢ Eq (HPow.hPow (gaussSum χ ψ) (orderOf χ)) (HMul.hMul (HMul.hMul (χ (-1)) ↑(F …
  -/
  apply_fun (gaussSum χ ψ * ·) at this
  /-
    F : Type u_1
    R : Type u_2
    inst✝³ : Fintype F
    inst✝² : Field F
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    χ : MulChar F R
    ψ : AddChar F R
    hχ : LE.le 2 (orderOf χ)
    hψ : ψ.IsPrimitive
    this : Eq (HMul.hMul (gaussSum χ ψ) (HPow.hPow (gaussSum χ ψ) (HSub.hSub (orde …
    ⊢ Eq (HPow.hPow (gaussSum χ ψ) (orderOf χ)) (HMul.hMul (HMul.hMul (χ (-1)) ↑(F …
  -/
  rw [← pow_succ', Nat.sub_one_add_one_eq_of_pos (by omega)] at this
  have hχ₁ : χ ≠ 1 :=
    fun h ↦ ((orderOf_one (G := MulChar F R) ▸ h ▸ hχ).trans_lt Nat.one_lt_two).false
  /-
    F : Type u_1
    R : Type u_2
    inst✝³ : Fintype F
    inst✝² : Field F
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    χ : MulChar F R
    ψ : AddChar F R
    hχ : LE.le 2 (orderOf χ)
    hψ : ψ.IsPrimitive
    this : Eq (HPow.hPow (gaussSum χ ψ) (orderOf χ)) (HMul.hMul (gaussSum χ ψ) (HM …
    hχ₁ : Ne χ 1
    ⊢ Eq (HPow.hPow (gaussSum χ ψ) (orderOf χ)) (HMul.hMul (HMul.hMul (χ (-1)) ↑(F …
  -/
  rw [this, ← mul_assoc, gaussSum_mul_gaussSum_pow_orderOf_sub_one hχ₁ hψ]
  /-
    🎉 no goals
  -/


