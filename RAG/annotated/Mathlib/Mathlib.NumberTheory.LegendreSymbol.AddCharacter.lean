/-- The values of an additive character on a ring of positive characteristic are roots of unity. -/
lemma val_mem_rootsOfUnity (φ : AddChar R R') (a : R) (h : 0 < ringChar R) :
    (φ.val_isUnit a).unit ∈ rootsOfUnity (ringChar R).toPNat' R' := by
  simp only [mem_rootsOfUnity', IsUnit.unit_spec, Nat.toPNat'_coe, h, ↓reduceIte,
    ← map_nsmul_eq_pow, nsmul_eq_mul, CharP.cast_eq_zero, zero_mul, map_zero_eq_one]


/-- An additive character is *primitive* iff all its multiplicative shifts by nonzero
elements are nontrivial. -/
def IsPrimitive (ψ : AddChar R R') : Prop := ∀ ⦃a : R⦄, a ≠ 0 → mulShift ψ a ≠ 1


/-- The composition of a primitive additive character with an injective mooid homomorphism
is also primitive. -/
lemma IsPrimitive.compMulHom_of_isPrimitive {R'' : Type*} [CommMonoid R''] {φ : AddChar R R'}
    {f : R' →* R''} (hφ : φ.IsPrimitive) (hf : Function.Injective f) :
    (f.compAddChar φ).IsPrimitive := fun a ha ↦ by
  /-
    R : Type u
    inst✝² : CommRing R
    R' : Type v
    inst✝¹ : CommMonoid R'
    R'' : Type u_1
    inst✝ : CommMonoid R''
    φ : AddChar R R'
    f : MonoidHom R' R''
    hφ : φ.IsPrimitive
    hf : Function.Injective ⇑f
    a : R
    ha : Ne a 0
    ⊢ Ne ((f.compAddChar φ).mulShift a) 1
  -/
  simpa [DFunLike.ext_iff] using (MonoidHom.compAddChar_injective_right f hf).ne (hφ ha)
  /-
    🎉 no goals
  -/


/-- The map associating to `a : R` the multiplicative shift of `ψ` by `a`
is injective when `ψ` is primitive. -/
theorem to_mulShift_inj_of_isPrimitive {ψ : AddChar R R'} (hψ : IsPrimitive ψ) :
    Function.Injective ψ.mulShift := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    R' : Type v
    inst✝ : CommMonoid R'
    ψ : AddChar R R'
    hψ : ψ.IsPrimitive
    ⊢ Function.Injective ψ.mulShift
  -/
  intro a b h
  /-
    R : Type u
    inst✝¹ : CommRing R
    R' : Type v
    inst✝ : CommMonoid R'
    ψ : AddChar R R'
    hψ : ψ.IsPrimitive
    a b : R
    h : Eq (ψ.mulShift a) (ψ.mulShift b)
    ⊢ Eq a b
  -/
  apply_fun fun x => x * mulShift ψ (-b) at h
  /-
    R : Type u
    inst✝¹ : CommRing R
    R' : Type v
    inst✝ : CommMonoid R'
    ψ : AddChar R R'
    hψ : ψ.IsPrimitive
    a b : R
    h : Eq (HMul.hMul (ψ.mulShift a) (ψ.mulShift (Neg.neg b))) (HMul.hMul (ψ.mulSh …
    ⊢ Eq a b
  -/
  simp only [mulShift_mul, mulShift_zero, add_neg_cancel, mulShift_apply] at h
  /-
    R : Type u
    inst✝¹ : CommRing R
    R' : Type v
    inst✝ : CommMonoid R'
    ψ : AddChar R R'
    hψ : ψ.IsPrimitive
    a b : R
    h : Eq (ψ.mulShift (HAdd.hAdd a (Neg.neg b))) 1
    ⊢ Eq a b
  -/
  simpa [← sub_eq_add_neg, sub_eq_zero] using (hψ · h)
  /-
    🎉 no goals
  -/

-- `AddCommGroup.equiv_direct_sum_zmod_of_fintype`
-- gives the structure theorem for finite abelian groups.
-- This could be used to show that the map above is a bijection.
-- We leave this for a later occasion.

/-- When `R` is a field `F`, then a nontrivial additive character is primitive -/
theorem IsPrimitive.of_ne_one {F : Type u} [Field F] {ψ : AddChar F R'} (hψ : ψ ≠ 1) :
    IsPrimitive ψ :=
                        /-
                          R' : Type v
                          inst✝¹ : CommMonoid R'
                          F : Type u
                          inst✝ : Field F
                          ψ : AddChar F R'
                          hψ : Ne ψ 1
                          a : F
                          ha : Ne a 0
                          h : Eq (ψ.mulShift a) 1
                          ⊢ Eq ψ 1
                        -/
  fun a ha h ↦ hψ <| by simpa [mulShift_mulShift, ha] using congr_arg (mulShift · a⁻¹) h
                        /-
                          🎉 no goals
                        -/


/-- If `r` is not a unit, then `e.mulShift r` is not primitive. -/
lemma not_isPrimitive_mulShift [Finite R] (e : AddChar R R') {r : R}
    (hr : ¬ IsUnit r) : ¬ IsPrimitive (e.mulShift r) := by
  /-
    R : Type u
    inst✝² : CommRing R
    R' : Type v
    inst✝¹ : CommMonoid R'
    inst✝ : Finite R
    e : AddChar R R'
    r : R
    hr : Not (IsUnit r)
    ⊢ Not (e.mulShift r).IsPrimitive
  -/
  simp only [IsPrimitive, not_forall]
  /-
    R : Type u
    inst✝² : CommRing R
    R' : Type v
    inst✝¹ : CommMonoid R'
    inst✝ : Finite R
    e : AddChar R R'
    r : R
    hr : Not (IsUnit r)
    ⊢ Exists fun x => Exists fun x_1 => Not (Ne ((e.mulShift r).mulShift x) 1)
  -/
  simp only [isUnit_iff_mem_nonZeroDivisors_of_finite, mem_nonZeroDivisors_iff, not_forall] at hr
  /-
    R : Type u
    inst✝² : CommRing R
    R' : Type v
    inst✝¹ : CommMonoid R'
    inst✝ : Finite R
    e : AddChar R R'
    r : R
    hr : Exists fun x => Exists fun x_1 => Not (Eq x 0)
    ⊢ Exists fun x => Exists fun x_1 => Not (Ne ((e.mulShift r).mulShift x) 1)
  -/
  rcases hr with ⟨x, h, h'⟩
  /-
    case intro.intro
    R : Type u
    inst✝² : CommRing R
    R' : Type v
    inst✝¹ : CommMonoid R'
    inst✝ : Finite R
    e : AddChar R R'
    r x : R
    h : Eq (HMul.hMul x r) 0
    h' : Not (Eq x 0)
    ⊢ Exists fun x => Exists fun x_1 => Not (Ne ((e.mulShift r).mulShift x) 1)
  -/
  exact ⟨x, h', by simp only [mulShift_mulShift, mul_comm r, h, mulShift_zero, not_ne_iff]⟩
  /-
    🎉 no goals
  -/


/-- Definition for a primitive additive character on a finite ring `R` into a cyclotomic extension
of a field `R'`. It records which cyclotomic extension it is, the character, and the
fact that the character is primitive. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- can't prove that they always exist (referring to providing an `Inhabited` instance)
-- @[nolint has_nonempty_instance]
structure PrimitiveAddChar (R : Type u) [CommRing R] (R' : Type v) [Field R'] where
  /-- The first projection from `PrimitiveAddChar`, giving the cyclotomic field. -/
  n : ℕ+
  /-- The second projection from `PrimitiveAddChar`, giving the character. -/
  char : AddChar R (CyclotomicField n R')
  /-- The third projection from `PrimitiveAddChar`, showing that `χ.char` is primitive. -/
  prim : IsPrimitive char


/-- If `e` is not primitive, then `e.mulShift d = 1` for some proper divisor `d` of `N`. -/
lemma exists_divisor_of_not_isPrimitive (he : ¬e.IsPrimitive) :
    ∃ d : ℕ, d ∣ N ∧ d < N ∧ e.mulShift d = 1 := by
  /-
    N : Nat
    inst✝¹ : NeZero N
    R : Type u_1
    inst✝ : CommRing R
    e : AddChar (ZMod N) R
    he : Not e.IsPrimitive
    ⊢ Exists fun d => And (Dvd.dvd d N) (And (LT.lt d N) (Eq (e.mulShift ↑d) 1))
  -/
  simp_rw [IsPrimitive, not_forall, not_ne_iff] at he
  /-
    N : Nat
    inst✝¹ : NeZero N
    R : Type u_1
    inst✝ : CommRing R
    e : AddChar (ZMod N) R
    he : Exists fun x => Exists fun h => Eq (e.mulShift x) 1
    ⊢ Exists fun d => And (Dvd.dvd d N) (And (LT.lt d N) (Eq (e.mulShift ↑d) 1))
  -/
  rcases he with ⟨b, hb_ne, hb⟩
  -- We have `AddChar.mulShift e b = 1`, but `b ≠ 0`.
  /-
    case intro.intro
    N : Nat
    inst✝¹ : NeZero N
    R : Type u_1
    inst✝ : CommRing R
    e : AddChar (ZMod N) R
    b : ZMod N
    hb_ne : Ne b 0
    hb : Eq (e.mulShift b) 1
    ⊢ Exists fun d => And (Dvd.dvd d N) (And (LT.lt d N) (Eq (e.mulShift ↑d) 1))
  -/
  obtain ⟨d, hd, u, hu, rfl⟩ := b.eq_unit_mul_divisor
  /-
    case intro.intro.intro.intro.intro.intro
    N : Nat
    inst✝¹ : NeZero N
    R : Type u_1
    inst✝ : CommRing R
    e : AddChar (ZMod N) R
    d : Nat
    hd : Dvd.dvd d N
    u : ZMod N
    hu : IsUnit u
    hb_ne : Ne (HMul.hMul u ↑d) 0
    hb : Eq (e.mulShift (HMul.hMul u ↑d)) 1
    ⊢ Exists fun d => And (Dvd.dvd d N) (And (LT.lt d N) (Eq (e.mulShift ↑d) 1))
  -/
  refine ⟨d, hd, lt_of_le_of_ne (Nat.le_of_dvd (NeZero.pos _) hd) ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      N : Nat
      inst✝¹ : NeZero N
      R : Type u_1
      inst✝ : CommRing R
      e : AddChar (ZMod N) R
      d : Nat
      hd : Dvd.dvd d N
      u : ZMod N
      hu : IsUnit u
      hb_ne : Ne (HMul.hMul u ↑d) 0
      hb : Eq (e.mulShift (HMul.hMul u ↑d)) 1
      ⊢ Ne d N
    -/
  · exact fun h ↦ by simp only [h, ZMod.natCast_self, mul_zero, ne_eq, not_true_eq_false] at hb_ne
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      N : Nat
      inst✝¹ : NeZero N
      R : Type u_1
      inst✝ : CommRing R
      e : AddChar (ZMod N) R
      d : Nat
      hd : Dvd.dvd d N
      u : ZMod N
      hu : IsUnit u
      hb_ne : Ne (HMul.hMul u ↑d) 0
      hb : Eq (e.mulShift (HMul.hMul u ↑d)) 1
      ⊢ Eq (e.mulShift ↑d) 1
    -/
  · rw [← mulShift_unit_eq_one_iff _ hu, ← hb, mul_comm]
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      N : Nat
      inst✝¹ : NeZero N
      R : Type u_1
      inst✝ : CommRing R
      e : AddChar (ZMod N) R
      d : Nat
      hd : Dvd.dvd d N
      u : ZMod N
      hu : IsUnit u
      hb_ne : Ne (HMul.hMul u ↑d) 0
      hb : Eq (e.mulShift (HMul.hMul u ↑d)) 1
      ⊢ Eq ((e.mulShift ↑d).mulShift u) (e.mulShift (HMul.hMul (↑d) u))
    -/
    ext1 y
    /-
      case intro.intro.intro.intro.intro.intro.refine_2.h
      N : Nat
      inst✝¹ : NeZero N
      R : Type u_1
      inst✝ : CommRing R
      e : AddChar (ZMod N) R
      d : Nat
      hd : Dvd.dvd d N
      u : ZMod N
      hu : IsUnit u
      hb_ne : Ne (HMul.hMul u ↑d) 0
      hb : Eq (e.mulShift (HMul.hMul u ↑d)) 1
      y : ZMod N
      ⊢ Eq (((e.mulShift ↑d).mulShift u) y) ((e.mulShift (HMul.hMul (↑d) u)) y)
    -/
    rw [mulShift_apply, mulShift_apply, mulShift_apply, mul_assoc]
    /-
      🎉 no goals
    -/


/-- We can define an additive character on `ZMod n` when we have an `n`th root of unity `ζ : C`. -/
def zmodChar (n : ℕ) [NeZero n] {ζ : C} (hζ : ζ ^ n = 1) : AddChar (ZMod n) C where
  toFun a := ζ ^ a.val
                         /-
                           R : Type u
                           inst✝³ : CommRing R
                           R' : Type v
                           inst✝² : CommMonoid R'
                           C : Type v
                           inst✝¹ : CommMonoid C
                           n : Nat
                           inst✝ : NeZero n
                           ζ : C
                           hζ : Eq (HPow.hPow ζ n) 1
                           ⊢ Eq ((fun a => HPow.hPow ζ a.val) 0) 1
                         -/
  map_zero_eq_one' := by simp only [ZMod.val_zero, pow_zero]
                         /-
                           🎉 no goals
                         -/
                            /-
                              R : Type u
                              inst✝³ : CommRing R
                              R' : Type v
                              inst✝² : CommMonoid R'
                              C : Type v
                              inst✝¹ : CommMonoid C
                              n : Nat
                              inst✝ : NeZero n
                              ζ : C
                              hζ : Eq (HPow.hPow ζ n) 1
                              x y : ZMod n
                              ⊢ Eq ((fun a => HPow.hPow ζ a.val) (HAdd.hAdd x y)) (HMul.hMul ((fun a => HPow …
                            -/
  map_add_eq_mul' x y := by simp only [ZMod.val_add, ← pow_eq_pow_mod _ hζ, ← pow_add]
                            /-
                              🎉 no goals
                            -/


/-- The additive character on `ZMod n` defined using `ζ` sends `a` to `ζ^a`. -/
theorem zmodChar_apply {n : ℕ} [NeZero n] {ζ : C} (hζ : ζ ^ n = 1) (a : ZMod n) :
    zmodChar n hζ a = ζ ^ a.val :=
  rfl


theorem zmodChar_apply' {n : ℕ} [NeZero n] {ζ : C} (hζ : ζ ^ n = 1) (a : ℕ) :
    zmodChar n hζ a = ζ ^ a := by
  /-
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ζ : C
    hζ : Eq (HPow.hPow ζ n) 1
    a : Nat
    ⊢ Eq ((AddChar.zmodChar n hζ) ↑a) (HPow.hPow ζ a)
  -/
  rw [pow_eq_pow_mod a hζ, zmodChar_apply, ZMod.val_natCast a]
  /-
    🎉 no goals
  -/


/-- An additive character on `ZMod n` is nontrivial iff it takes a value `≠ 1` on `1`. -/
theorem zmod_char_ne_one_iff (n : ℕ) [NeZero n] (ψ : AddChar (ZMod n) C) : ψ ≠ 1 ↔ ψ 1 ≠ 1 := by
  /-
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ψ : AddChar (ZMod n) C
    ⊢ Iff (Ne ψ 1) (Ne (ψ 1) 1)
  -/
  rw [ne_one_iff]
  /-
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ψ : AddChar (ZMod n) C
    ⊢ Iff (Exists fun x => Ne (ψ x) 1) (Ne (ψ 1) 1)
  -/
  refine ⟨?_, fun h => ⟨_, h⟩⟩
  /-
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ψ : AddChar (ZMod n) C
    ⊢ (Exists fun x => Ne (ψ x) 1) → Ne (ψ 1) 1
  -/
  contrapose!
  /-
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ψ : AddChar (ZMod n) C
    ⊢ Eq (ψ 1) 1 → ∀ (x : ZMod n), Eq (ψ x) 1
  -/
  rintro h₁ a
  have ha₁ : a = a.val • (1 : ZMod ↑n) := by
    rw [nsmul_eq_mul, mul_one]; exact (ZMod.natCast_zmod_val a).symm
  /-
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ψ : AddChar (ZMod n) C
    h₁ : Eq (ψ 1) 1
    a : ZMod n
    ha₁ : Eq a (HSMul.hSMul a.val 1)
    ⊢ Eq (ψ a) 1
  -/
  rw [ha₁, map_nsmul_eq_pow, h₁, one_pow]
  /-
    🎉 no goals
  -/


/-- A primitive additive character on `ZMod n` takes the value `1` only at `0`. -/
theorem IsPrimitive.zmod_char_eq_one_iff (n : ℕ) [NeZero n]
    {ψ : AddChar (ZMod n) C} (hψ : IsPrimitive ψ) (a : ZMod n) :
    ψ a = 1 ↔ a = 0 := by
  /-
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ψ : AddChar (ZMod n) C
    hψ : ψ.IsPrimitive
    a : ZMod n
    ⊢ Iff (Eq (ψ a) 1) (Eq a 0)
  -/
  refine ⟨fun h => not_imp_comm.mp (@hψ a) ?_, fun ha => by rw [ha, map_zero_eq_one]⟩
  /-
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ψ : AddChar (ZMod n) C
    hψ : ψ.IsPrimitive
    a : ZMod n
    h : Eq (ψ a) 1
    ⊢ Not (Ne (ψ.mulShift a) 1)
  -/
  rw [zmod_char_ne_one_iff n (mulShift ψ a), mulShift_apply, mul_one, h, Classical.not_not]
  /-
    🎉 no goals
  -/


/-- The converse: if the additive character takes the value `1` only at `0`,
then it is primitive. -/
theorem zmod_char_primitive_of_eq_one_only_at_zero (n : ℕ) (ψ : AddChar (ZMod n) C)
    (hψ : ∀ a, ψ a = 1 → a = 0) : IsPrimitive ψ := by
  /-
    C : Type v
    inst✝ : CommMonoid C
    n : Nat
    ψ : AddChar (ZMod n) C
    hψ : ∀ (a : ZMod n), Eq (ψ a) 1 → Eq a 0
    ⊢ ψ.IsPrimitive
  -/
  refine fun a ha hf => ?_
  have h : mulShift ψ a 1 = (1 : AddChar (ZMod n) C) (1 : ZMod n) :=
    congr_fun (congr_arg (↑) hf) 1
  /-
    C : Type v
    inst✝ : CommMonoid C
    n : Nat
    ψ : AddChar (ZMod n) C
    hψ : ∀ (a : ZMod n), Eq (ψ a) 1 → Eq a 0
    a : ZMod n
    ha : Ne a 0
    hf : Eq (ψ.mulShift a) 1
    h : Eq ((ψ.mulShift a) 1) (1 1)
    ⊢ False
  -/
  rw [mulShift_apply, mul_one] at h; norm_cast at h
  /-
    C : Type v
    inst✝ : CommMonoid C
    n : Nat
    ψ : AddChar (ZMod n) C
    hψ : ∀ (a : ZMod n), Eq (ψ a) 1 → Eq a 0
    a : ZMod n
    ha : Ne a 0
    hf : Eq (ψ.mulShift a) 1
    h : Eq (ψ a) (1 1)
    ⊢ False
  -/
  exact ha (hψ a h)
  /-
    🎉 no goals
  -/


/-- The additive character on `ZMod n` associated to a primitive `n`th root of unity
is primitive -/
theorem zmodChar_primitive_of_primitive_root (n : ℕ) [NeZero n] {ζ : C} (h : IsPrimitiveRoot ζ n) :
    IsPrimitive (zmodChar n ((IsPrimitiveRoot.iff_def ζ n).mp h).left) := by
  /-
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ζ : C
    h : IsPrimitiveRoot ζ n
    ⊢ (AddChar.zmodChar n ⋯).IsPrimitive
  -/
  apply zmod_char_primitive_of_eq_one_only_at_zero
  /-
    case hψ
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ζ : C
    h : IsPrimitiveRoot ζ n
    ⊢ ∀ (a : ZMod n), Eq ((AddChar.zmodChar n ⋯) a) 1 → Eq a 0
  -/
  intro a ha
  /-
    case hψ
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ζ : C
    h : IsPrimitiveRoot ζ n
    a : ZMod n
    ha : Eq ((AddChar.zmodChar n ⋯) a) 1
    ⊢ Eq a 0
  -/
  rw [zmodChar_apply, ← pow_zero ζ] at ha
  /-
    case hψ
    C : Type v
    inst✝¹ : CommMonoid C
    n : Nat
    inst✝ : NeZero n
    ζ : C
    h : IsPrimitiveRoot ζ n
    a : ZMod n
    ha : Eq (HPow.hPow ζ a.val) (HPow.hPow ζ 0)
    ⊢ Eq a 0
  -/
  exact (ZMod.val_eq_zero a).mp (IsPrimitiveRoot.pow_inj h (ZMod.val_lt a) (NeZero.pos _) ha)
  /-
    🎉 no goals
  -/


/-- There is a primitive additive character on `ZMod n` if the characteristic of the target
does not divide `n` -/
noncomputable def primitiveZModChar (n : ℕ+) (F' : Type v) [Field F'] (h : (n : F') ≠ 0) :
    PrimitiveAddChar (ZMod n) F' :=
  have : NeZero (n : F') := ⟨h⟩
  ⟨n, zmodChar n (IsCyclotomicExtension.zeta_pow n F' _),
    zmodChar_primitive_of_primitive_root n (IsCyclotomicExtension.zeta_spec n F' _)⟩


/-- There is a primitive additive character on the finite field `F` if the characteristic
of the target is different from that of `F`.

We obtain it as the composition of the trace from `F` to `ZMod p` with a primitive
additive character on `ZMod p`, where `p` is the characteristic of `F`. -/
noncomputable def FiniteField.primitiveChar (F F' : Type*) [Field F] [Finite F] [Field F']
    (h : ringChar F' ≠ ringChar F) : PrimitiveAddChar F F' := by
  /-
    F : Type u_1
    F' : Type u_2
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    ⊢ AddChar.PrimitiveAddChar F F'
  -/
  let p := ringChar F
  /-
    F : Type u_1
    F' : Type u_2
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    p : Nat := ringChar F
    ⊢ AddChar.PrimitiveAddChar F F'
  -/
  haveI hp : Fact p.Prime := ⟨CharP.char_is_prime F _⟩
  /-
    F : Type u_1
    F' : Type u_2
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    p : Nat := ringChar F
    hp : Fact (Nat.Prime p)
    ⊢ AddChar.PrimitiveAddChar F F'
  -/
  let pp := p.toPNat hp.1.pos
  have hp₂ : ¬ringChar F' ∣ p := by
    cases' CharP.char_is_prime_or_zero F' (ringChar F') with hq hq
    · exact mt (Nat.Prime.dvd_iff_eq hp.1 (Nat.Prime.ne_one hq)).mp h.symm
    · rw [hq]
      exact fun hf => Nat.Prime.ne_zero hp.1 (zero_dvd_iff.mp hf)
  /-
    F : Type u_1
    F' : Type u_2
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    p : Nat := ringChar F
    hp : Fact (Nat.Prime p)
    pp : PNat := p.toPNat ⋯
    hp₂ : Not (Dvd.dvd (ringChar F') p)
    ⊢ AddChar.PrimitiveAddChar F F'
  -/
  let ψ := primitiveZModChar pp F' (neZero_iff.mp (NeZero.of_not_dvd F' hp₂))
  /-
    F : Type u_1
    F' : Type u_2
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    p : Nat := ringChar F
    hp : Fact (Nat.Prime p)
    pp : PNat := p.toPNat ⋯
    hp₂ : Not (Dvd.dvd (ringChar F') p)
    ψ : AddChar.PrimitiveAddChar (ZMod ↑pp) F' := AddChar.primitiveZModChar pp F' ⋯
    ⊢ AddChar.PrimitiveAddChar F F'
  -/
  letI : Algebra (ZMod p) F := ZMod.algebra _ _
  /-
    F : Type u_1
    F' : Type u_2
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    p : Nat := ringChar F
    hp : Fact (Nat.Prime p)
    pp : PNat := p.toPNat ⋯
    hp₂ : Not (Dvd.dvd (ringChar F') p)
    ψ : AddChar.PrimitiveAddChar (ZMod ↑pp) F' := AddChar.primitiveZModChar pp F' ⋯
    this : Algebra (ZMod p) F := ZMod.algebra F p
    ⊢ AddChar.PrimitiveAddChar F F'
  -/
  let ψ' := ψ.char.compAddMonoidHom (Algebra.trace (ZMod p) F).toAddMonoidHom
  have hψ' : ψ' ≠ 1 := by
    obtain ⟨a, ha⟩ := FiniteField.trace_to_zmod_nondegenerate F one_ne_zero
    rw [one_mul] at ha
    exact ne_one_iff.2
      ⟨a, fun hf => ha <| (ψ.prim.zmod_char_eq_one_iff pp <| Algebra.trace (ZMod p) F a).mp hf⟩
  /-
    F : Type u_1
    F' : Type u_2
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Field F'
    h : Ne (ringChar F') (ringChar F)
    p : Nat := ringChar F
    hp : Fact (Nat.Prime p)
    pp : PNat := p.toPNat ⋯
    hp₂ : Not (Dvd.dvd (ringChar F') p)
    ψ : AddChar.PrimitiveAddChar (ZMod ↑pp) F' := AddChar.primitiveZModChar pp F' ⋯
    this : Algebra (ZMod p) F := ZMod.algebra F p
    ψ' : AddChar F (CyclotomicField ψ.n F') := ψ.char.compAddMonoidHom (Algebra.tr …
    hψ' : Ne ψ' 1
    ⊢ AddChar.PrimitiveAddChar F F'
  -/
  exact ⟨ψ.n, ψ', IsPrimitive.of_ne_one hψ'⟩
  /-
    🎉 no goals
  -/

@[deprecated (since := "2024-05-30")] alias primitiveCharFiniteField := FiniteField.primitiveChar


/-- The sum over the values of a nontrivial additive character vanishes if the target ring
is a domain. -/
theorem sum_eq_zero_of_ne_one [IsDomain R'] {ψ : AddChar R R'} (hψ : ψ ≠ 1) : ∑ a, ψ a = 0 := by
  /-
    R : Type u_1
    inst✝³ : AddGroup R
    inst✝² : Fintype R
    R' : Type u_2
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    ψ : AddChar R R'
    hψ : Ne ψ 1
    ⊢ Eq (Finset.univ.sum fun a => ψ a) 0
  -/
  rcases ne_one_iff.1 hψ with ⟨b, hb⟩
  have h₁ : ∑ a : R, ψ (b + a) = ∑ a : R, ψ a :=
    Fintype.sum_bijective _ (AddGroup.addLeft_bijective b) _ _ fun x => rfl
  /-
    case intro
    R : Type u_1
    inst✝³ : AddGroup R
    inst✝² : Fintype R
    R' : Type u_2
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    ψ : AddChar R R'
    hψ : Ne ψ 1
    b : R
    hb : Ne (ψ b) 1
    h₁ : Eq (Finset.univ.sum fun a => ψ (HAdd.hAdd b a)) (Finset.univ.sum fun a => …
    ⊢ Eq (Finset.univ.sum fun a => ψ a) 0
  -/
  simp_rw [map_add_eq_mul] at h₁
  /-
    case intro
    R : Type u_1
    inst✝³ : AddGroup R
    inst✝² : Fintype R
    R' : Type u_2
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    ψ : AddChar R R'
    hψ : Ne ψ 1
    b : R
    hb : Ne (ψ b) 1
    h₁ : Eq (Finset.univ.sum fun x => HMul.hMul (ψ b) (ψ x)) (Finset.univ.sum fun  …
    ⊢ Eq (Finset.univ.sum fun a => ψ a) 0
  -/
  have h₂ : ∑ a : R, ψ a = Finset.univ.sum ↑ψ := rfl
  /-
    case intro
    R : Type u_1
    inst✝³ : AddGroup R
    inst✝² : Fintype R
    R' : Type u_2
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    ψ : AddChar R R'
    hψ : Ne ψ 1
    b : R
    hb : Ne (ψ b) 1
    h₁ : Eq (Finset.univ.sum fun x => HMul.hMul (ψ b) (ψ x)) (Finset.univ.sum fun  …
    h₂ : Eq (Finset.univ.sum fun a => ψ a) (Finset.univ.sum ⇑ψ)
    ⊢ Eq (Finset.univ.sum fun a => ψ a) 0
  -/
  rw [← Finset.mul_sum, h₂] at h₁
  /-
    case intro
    R : Type u_1
    inst✝³ : AddGroup R
    inst✝² : Fintype R
    R' : Type u_2
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    ψ : AddChar R R'
    hψ : Ne ψ 1
    b : R
    hb : Ne (ψ b) 1
    h₁ : Eq (HMul.hMul (ψ b) (Finset.univ.sum ⇑ψ)) (Finset.univ.sum ⇑ψ)
    h₂ : Eq (Finset.univ.sum fun a => ψ a) (Finset.univ.sum ⇑ψ)
    ⊢ Eq (Finset.univ.sum fun a => ψ a) 0
  -/
  exact eq_zero_of_mul_eq_self_left hb h₁
  /-
    🎉 no goals
  -/


/-- The sum over the values of the trivial additive character is the cardinality of the source. -/
theorem sum_eq_card_of_eq_one {ψ : AddChar R R'} (hψ : ψ = 1) :
                                    /-
                                      R : Type u_1
                                      inst✝² : AddGroup R
                                      inst✝¹ : Fintype R
                                      R' : Type u_2
                                      inst✝ : CommRing R'
                                      ψ : AddChar R R'
                                      hψ : Eq ψ 1
                                      ⊢ Eq (Finset.univ.sum fun a => ψ a) ↑(Fintype.card R)
                                    -/
    ∑ a, ψ a = Fintype.card R := by simp [hψ]
                                    /-
                                      🎉 no goals
                                    -/


/-- The sum over the values of `mulShift ψ b` for `ψ` primitive is zero when `b ≠ 0`
and `#R` otherwise. -/
theorem sum_mulShift {R : Type*} [CommRing R] [Fintype R] [DecidableEq R]
    {R' : Type*} [CommRing R'] [IsDomain R'] {ψ : AddChar R R'} (b : R)
    (hψ : IsPrimitive ψ) : ∑ x : R, ψ (x * b) = if b = 0 then Fintype.card R else 0 := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : Fintype R
    inst✝² : DecidableEq R
    R' : Type u_2
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    ψ : AddChar R R'
    b : R
    hψ : ψ.IsPrimitive
    ⊢ Eq (Finset.univ.sum fun x => ψ (HMul.hMul x b)) ↑(ite (Eq b 0) (Fintype.card …
  -/
  split_ifs with h
  · -- case `b = 0`
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : Fintype R
      inst✝² : DecidableEq R
      R' : Type u_2
      inst✝¹ : CommRing R'
      inst✝ : IsDomain R'
      ψ : AddChar R R'
      b : R
      hψ : ψ.IsPrimitive
      h : Eq b 0
      ⊢ Eq (Finset.univ.sum fun x => ψ (HMul.hMul x b)) ↑(Fintype.card R)
    -/
    simp only [h, mul_zero, map_zero_eq_one, Finset.sum_const, Nat.smul_one_eq_cast]
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : Fintype R
      inst✝² : DecidableEq R
      R' : Type u_2
      inst✝¹ : CommRing R'
      inst✝ : IsDomain R'
      ψ : AddChar R R'
      b : R
      hψ : ψ.IsPrimitive
      h : Eq b 0
      ⊢ Eq ↑Finset.univ.card ↑(Fintype.card R)
    -/
    rfl
    /-
      🎉 no goals
    -/
  · -- case `b ≠ 0`
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : Fintype R
      inst✝² : DecidableEq R
      R' : Type u_2
      inst✝¹ : CommRing R'
      inst✝ : IsDomain R'
      ψ : AddChar R R'
      b : R
      hψ : ψ.IsPrimitive
      h : Not (Eq b 0)
      ⊢ Eq (Finset.univ.sum fun x => ψ (HMul.hMul x b)) ↑0
    -/
    simp_rw [mul_comm]
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : Fintype R
      inst✝² : DecidableEq R
      R' : Type u_2
      inst✝¹ : CommRing R'
      inst✝ : IsDomain R'
      ψ : AddChar R R'
      b : R
      hψ : ψ.IsPrimitive
      h : Not (Eq b 0)
      ⊢ Eq (Finset.univ.sum fun x => ψ (HMul.hMul b x)) ↑0
    -/
    exact mod_cast sum_eq_zero_of_ne_one (hψ h)
    /-
      🎉 no goals
    -/


/-- Post-composing an additive character to `ℂ` with complex conjugation gives the inverse
character. -/
lemma starComp_eq_inv (hR : 0 < ringChar R) {φ : AddChar R ℂ} :
    (starRingEnd ℂ).compAddChar φ = φ⁻¹ := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : LT.lt 0 (ringChar R)
    φ : AddChar R Complex
    ⊢ Eq ((↑(starRingEnd Complex)).compAddChar φ) (Inv.inv φ)
  -/
  ext1 a
  simp only [RingHom.toMonoidHom_eq_coe, MonoidHom.coe_compAddChar, MonoidHom.coe_coe,
    Function.comp_apply, inv_apply']
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    hR : LT.lt 0 (ringChar R)
    φ : AddChar R Complex
    a : R
    ⊢ Eq ((starRingEnd Complex) (φ a)) (Inv.inv (φ a))
  -/
  have H := Complex.norm_eq_one_of_mem_rootsOfUnity <| φ.val_mem_rootsOfUnity a hR
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    hR : LT.lt 0 (ringChar R)
    φ : AddChar R Complex
    a : R
    H : Eq (Norm.norm ↑⋯.unit) 1
    ⊢ Eq ((starRingEnd Complex) (φ a)) (Inv.inv (φ a))
  -/
  exact (Complex.inv_eq_conj H).symm
  /-
    🎉 no goals
  -/


lemma starComp_apply (hR : 0 < ringChar R) {φ : AddChar R ℂ} (a : R) :
    (starRingEnd ℂ) (φ a) = φ⁻¹ a := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : LT.lt 0 (ringChar R)
    φ : AddChar R Complex
    a : R
    ⊢ Eq ((starRingEnd Complex) (φ a)) ((Inv.inv φ) a)
  -/
  rw [← starComp_eq_inv hR]
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : LT.lt 0 (ringChar R)
    φ : AddChar R Complex
    a : R
    ⊢ Eq ((starRingEnd Complex) (φ a)) (((↑(starRingEnd Complex)).compAddChar φ) a)
  -/
  rfl
  /-
    🎉 no goals
  -/


private lemma ringChar_ne : ringChar ℂ ≠ ringChar F := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : Finite F
    ⊢ Ne (ringChar Complex) (ringChar F)
  -/
  simpa only [ringChar.eq_zero] using (CharP.ringChar_ne_zero_of_finite F).symm
  /-
    🎉 no goals
  -/


/--  A primitive additive character on the finite field `F` with values in `ℂ`. -/
noncomputable def FiniteField.primitiveChar_to_Complex : AddChar F ℂ := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : Finite F
    ⊢ AddChar F Complex
  -/
  refine MonoidHom.compAddChar ?_ (primitiveChar F ℂ <| ringChar_ne F).char
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : Finite F
    ⊢ MonoidHom (CyclotomicField (AddChar.FiniteField.primitiveChar F Complex ⋯).n …
  -/
  exact (IsCyclotomicExtension.algEquiv ?n ℂ (CyclotomicField ?n ℂ) ℂ : CyclotomicField ?n ℂ →* ℂ)
  /-
    🎉 no goals
  -/


lemma FiniteField.primitiveChar_to_Complex_isPrimitive :
    (primitiveChar_to_Complex F).IsPrimitive := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : Finite F
    ⊢ (AddChar.FiniteField.primitiveChar_to_Complex F).IsPrimitive
  -/
  refine IsPrimitive.compMulHom_of_isPrimitive (PrimitiveAddChar.prim _) ?_
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : Finite F
    ⊢ Function.Injective ⇑↑(IsCyclotomicExtension.algEquiv (AddChar.FiniteField.pr …
  -/
  let nn := (primitiveChar F ℂ <| ringChar_ne F).n
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : Finite F
    nn : PNat := (AddChar.FiniteField.primitiveChar F Complex ⋯).n
    ⊢ Function.Injective ⇑↑(IsCyclotomicExtension.algEquiv (AddChar.FiniteField.pr …
  -/
  exact (IsCyclotomicExtension.algEquiv nn ℂ (CyclotomicField nn ℂ) ℂ).injective
  /-
    🎉 no goals
  -/


