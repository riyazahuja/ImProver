/-- `AddChar A M` is the type of maps `A → M`, for `A` an additive monoid and `M` a multiplicative
monoid, which intertwine addition in `A` with multiplication in `M`.

We only put the typeclasses needed for the definition, although in practice we are usually
interested in much more specific cases (e.g. when `A` is a group and `M` a commutative ring).
 -/
structure AddChar where
  /-- The underlying function.

  Do not use this function directly. Instead use the coercion coming from the `FunLike`
  instance. -/
  toFun : A → M
  /-- The function maps `0` to `1`.

  Do not use this directly. Instead use `AddChar.map_zero_eq_one`. -/
  map_zero_eq_one' : toFun 0 = 1
  /-- The function maps addition in `A` to multiplication in `M`.

  Do not use this directly. Instead use `AddChar.map_add_eq_mul`. -/
  map_add_eq_mul' : ∀ a b : A, toFun (a + b) = toFun a * toFun b


/-- Define coercion to a function. -/
instance instFunLike : FunLike (AddChar A M) A M where
  coe := AddChar.toFun
                             /-
                               A : Type u_1
                               B : Type u_2
                               M : Type u_3
                               N : Type u_4
                               inst✝³ : AddMonoid A
                               inst✝² : AddMonoid B
                               inst✝¹ : Monoid M
                               inst✝ : Monoid N
                               ψ✝ φ ψ : AddChar A M
                               h : Eq φ.toFun ψ.toFun
                               ⊢ Eq φ ψ
                             -/
  coe_injective' φ ψ h := by cases φ; cases ψ; congr
                                               /-
                                                 🎉 no goals
                                               -/


@[ext] lemma ext (f g : AddChar A M) (h : ∀ x : A, f x = g x) : f = g :=
  DFunLike.ext f g h


@[simp] lemma coe_mk (f : A → M)
    (map_zero_eq_one' : f 0 = 1) (map_add_eq_mul' : ∀ a b : A, f (a + b) = f a * f b) :
    AddChar.mk f map_zero_eq_one' map_add_eq_mul' = f := by
  /-
    A : Type u_1
    M : Type u_3
    inst✝¹ : AddMonoid A
    inst✝ : Monoid M
    f : A → M
    map_zero_eq_one' : Eq (f 0) 1
    map_add_eq_mul' : ∀ (a b : A), Eq (f (HAdd.hAdd a b)) (HMul.hMul (f a) (f b))
    ⊢ Eq (⇑{ toFun := f, map_zero_eq_one' := map_zero_eq_one', map_add_eq_mul' :=  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- An additive character maps `0` to `1`. -/
@[simp] lemma map_zero_eq_one (ψ : AddChar A M) : ψ 0 = 1 := ψ.map_zero_eq_one'


/-- An additive character maps sums to products. -/
lemma map_add_eq_mul (ψ : AddChar A M) (x y : A) : ψ (x + y) = ψ x * ψ y := ψ.map_add_eq_mul' x y


@[deprecated (since := "2024-06-06")] alias map_zero_one := map_zero_eq_one

@[deprecated (since := "2024-06-06")] alias map_add_mul := map_add_eq_mul


/-- Interpret an additive character as a monoid homomorphism. -/
def toMonoidHom (φ : AddChar A M) : Multiplicative A →* M where
  toFun := φ.toFun
  map_one' := φ.map_zero_eq_one'
  map_mul' := φ.map_add_eq_mul'

-- this instance was a bad idea and conflicted with `instFunLike` above


@[simp] lemma toMonoidHom_apply (ψ : AddChar A M) (a : Multiplicative A) :
  ψ.toMonoidHom a = ψ a.toAdd :=
  rfl


/-- An additive character maps multiples by natural numbers to powers. -/
lemma map_nsmul_eq_pow (ψ : AddChar A M) (n : ℕ) (x : A) : ψ (n • x) = ψ x ^ n :=
  ψ.toMonoidHom.map_pow x n


@[deprecated (since := "2024-06-06")] alias map_nsmul_pow := map_nsmul_eq_pow


/-- Additive characters `A → M` are the same thing as monoid homomorphisms from `Multiplicative A`
to `M`. -/
def toMonoidHomEquiv : AddChar A M ≃ (Multiplicative A →* M) where
  toFun φ := φ.toMonoidHom
  invFun f :=
  { toFun := f.toFun
    map_zero_eq_one' := f.map_one'
    map_add_eq_mul' := f.map_mul' }
  left_inv _ := rfl
  right_inv _ := rfl


@[simp, norm_cast] lemma coe_toMonoidHomEquiv (ψ : AddChar A M) :
    ⇑(toMonoidHomEquiv ψ) = ψ ∘ Multiplicative.toAdd := rfl


@[simp, norm_cast] lemma coe_toMonoidHomEquiv_symm (ψ : Multiplicative A →* M) :
    ⇑(toMonoidHomEquiv.symm ψ) = ψ ∘ Multiplicative.ofAdd := rfl


@[simp] lemma toMonoidHomEquiv_apply (ψ : AddChar A M) (a : Multiplicative A) :
    toMonoidHomEquiv ψ a = ψ a.toAdd := rfl


@[simp] lemma toMonoidHomEquiv_symm_apply (ψ : Multiplicative A →* M) (a : A) :
    toMonoidHomEquiv.symm ψ a = ψ (Multiplicative.ofAdd a) := rfl


/-- Interpret an additive character as a monoid homomorphism. -/
def toAddMonoidHom (φ : AddChar A M) : A →+ Additive M where
  toFun := φ.toFun
  map_zero' := φ.map_zero_eq_one'
  map_add' := φ.map_add_eq_mul'


@[simp] lemma coe_toAddMonoidHom (ψ : AddChar A M) : ⇑ψ.toAddMonoidHom = Additive.ofMul ∘ ψ := rfl


@[simp] lemma toAddMonoidHom_apply (ψ : AddChar A M) (a : A) :
    ψ.toAddMonoidHom a = Additive.ofMul (ψ a) := rfl


/-- Additive characters `A → M` are the same thing as additive homomorphisms from `A` to
`Additive M`. -/
def toAddMonoidHomEquiv : AddChar A M ≃ (A →+ Additive M) where
  toFun φ := φ.toAddMonoidHom
  invFun f :=
  { toFun := f.toFun
    map_zero_eq_one' := f.map_zero'
    map_add_eq_mul' := f.map_add' }
  left_inv _ := rfl
  right_inv _ := rfl


@[simp, norm_cast]
lemma coe_toAddMonoidHomEquiv (ψ : AddChar A M) :
    ⇑(toAddMonoidHomEquiv ψ) = Additive.ofMul ∘ ψ := rfl


@[simp, norm_cast] lemma coe_toAddMonoidHomEquiv_symm (ψ : A →+ Additive M) :
    ⇑(toAddMonoidHomEquiv.symm ψ) = Additive.toMul ∘ ψ := rfl


@[simp] lemma toAddMonoidHomEquiv_apply (ψ : AddChar A M) (a : A) :
    toAddMonoidHomEquiv ψ a = Additive.ofMul (ψ a) := rfl


@[simp] lemma toAddMonoidHomEquiv_symm_apply (ψ : A →+ Additive M) (a : A) :
    toAddMonoidHomEquiv.symm ψ a = (ψ a).toMul  := rfl


/-- The trivial additive character (sending everything to `1`). -/
instance instOne : One (AddChar A M) := toMonoidHomEquiv.one


/-- The trivial additive character (sending everything to `1`). -/
instance instZero : Zero (AddChar A M) := ⟨1⟩


@[simp, norm_cast] lemma coe_one : ⇑(1 : AddChar A M) = 1 := rfl

@[simp, norm_cast] lemma coe_zero : ⇑(0 : AddChar A M) = 1 := rfl

@[simp] lemma one_apply (a : A) : (1 : AddChar A M) a = 1 := rfl

@[simp] lemma zero_apply (a : A) : (0 : AddChar A M) a = 1 := rfl


lemma one_eq_zero : (1 : AddChar A M) = (0 : AddChar A M) := rfl


                                                           /-
                                                             A : Type u_1
                                                             M : Type u_3
                                                             inst✝¹ : AddMonoid A
                                                             inst✝ : Monoid M
                                                             ψ : AddChar A M
                                                             ⊢ Iff (Eq (⇑ψ) 1) (Eq ψ 0)
                                                           -/
@[simp, norm_cast] lemma coe_eq_one : ⇑ψ = 1 ↔ ψ = 0 := by rw [← coe_zero, DFunLike.coe_fn_eq]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp] lemma toMonoidHomEquiv_zero : toMonoidHomEquiv (0 : AddChar A M) = 1 := rfl

@[simp] lemma toMonoidHomEquiv_symm_one :
    toMonoidHomEquiv.symm (1 : Multiplicative A →* M) = 0 := rfl


@[simp] lemma toAddMonoidHomEquiv_zero : toAddMonoidHomEquiv (0 : AddChar A M) = 0 := rfl

@[simp] lemma toAddMonoidHomEquiv_symm_zero :
    toAddMonoidHomEquiv.symm (0 : A →+ Additive M) = 0 := rfl


instance instInhabited : Inhabited (AddChar A M) := ⟨1⟩


/-- Composing a `MonoidHom` with an `AddChar` yields another `AddChar`. -/
def _root_.MonoidHom.compAddChar {N : Type*} [Monoid N] (f : M →* N) (φ : AddChar A M) :
    AddChar A N := toMonoidHomEquiv.symm (f.comp φ.toMonoidHom)


@[simp, norm_cast]
lemma _root_.MonoidHom.coe_compAddChar {N : Type*} [Monoid N] (f : M →* N) (φ : AddChar A M) :
    f.compAddChar φ = f ∘ φ :=
  rfl


@[simp, norm_cast]
lemma _root_.MonoidHom.compAddChar_apply (f : M →* N) (φ : AddChar A M) : f.compAddChar φ = f ∘ φ :=
  rfl


lemma _root_.MonoidHom.compAddChar_injective_left (ψ : AddChar A M) (hψ : Surjective ψ) :
    Injective fun f : M →* N ↦ f.compAddChar ψ := by
  /-
    A : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝² : AddMonoid A
    inst✝¹ : Monoid M
    inst✝ : Monoid N
    ψ : AddChar A M
    hψ : Function.Surjective ⇑ψ
    ⊢ Function.Injective fun f => f.compAddChar ψ
  -/
  rintro f g h; rw [DFunLike.ext'_iff] at h ⊢; exact hψ.injective_comp_right h
                                               /-
                                                 🎉 no goals
                                               -/


lemma _root_.MonoidHom.compAddChar_injective_right (f : M →* N) (hf : Injective f) :
    Injective fun ψ : AddChar B M ↦ f.compAddChar ψ := by
  /-
    B : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝² : AddMonoid B
    inst✝¹ : Monoid M
    inst✝ : Monoid N
    f : MonoidHom M N
    hf : Function.Injective ⇑f
    ⊢ Function.Injective fun ψ => f.compAddChar ψ
  -/
  rintro ψ χ h; rw [DFunLike.ext'_iff] at h ⊢; exact hf.comp_left h
                                               /-
                                                 🎉 no goals
                                               -/


/-- Composing an `AddChar` with an `AddMonoidHom` yields another `AddChar`. -/
def compAddMonoidHom (φ : AddChar B M) (f : A →+ B) : AddChar A M :=
  toAddMonoidHomEquiv.symm (φ.toAddMonoidHom.comp f)


@[simp, norm_cast]
lemma coe_compAddMonoidHom (φ : AddChar B M) (f : A →+ B) : φ.compAddMonoidHom f = φ ∘ f := rfl


@[simp] lemma compAddMonoidHom_apply (ψ : AddChar B M) (f : A →+ B)
    (a : A) : ψ.compAddMonoidHom f a = ψ (f a) := rfl


lemma compAddMonoidHom_injective_left (f : A →+ B) (hf : Surjective f) :
    Injective fun ψ : AddChar B M ↦ ψ.compAddMonoidHom f := by
  /-
    A : Type u_1
    B : Type u_2
    M : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    inst✝ : Monoid M
    f : AddMonoidHom A B
    hf : Function.Surjective ⇑f
    ⊢ Function.Injective fun ψ => ψ.compAddMonoidHom f
  -/
  rintro ψ χ h; rw [DFunLike.ext'_iff] at h ⊢; exact hf.injective_comp_right h
                                               /-
                                                 🎉 no goals
                                               -/


lemma compAddMonoidHom_injective_right (ψ : AddChar B M) (hψ : Injective ψ) :
    Injective fun f : A →+ B ↦ ψ.compAddMonoidHom f := by
  /-
    A : Type u_1
    B : Type u_2
    M : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    inst✝ : Monoid M
    ψ : AddChar B M
    hψ : Function.Injective ⇑ψ
    ⊢ Function.Injective fun f => ψ.compAddMonoidHom f
  -/
  rintro f g h
  /-
    A : Type u_1
    B : Type u_2
    M : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    inst✝ : Monoid M
    ψ : AddChar B M
    hψ : Function.Injective ⇑ψ
    f g : AddMonoidHom A B
    h : Eq ((fun f => ψ.compAddMonoidHom f) f) ((fun f => ψ.compAddMonoidHom f) g)
    ⊢ Eq f g
  -/
  rw [DFunLike.ext'_iff] at h ⊢; exact hψ.comp_left h
                                 /-
                                   🎉 no goals
                                 -/


lemma eq_one_iff : ψ = 1 ↔ ∀ x, ψ x = 1 := DFunLike.ext_iff

lemma eq_zero_iff : ψ = 0 ↔ ∀ x, ψ x = 1 := DFunLike.ext_iff

lemma ne_one_iff : ψ ≠ 1 ↔ ∃ x, ψ x ≠ 1 := DFunLike.ne_iff

lemma ne_zero_iff : ψ ≠ 0 ↔ ∃ x, ψ x ≠ 1 := DFunLike.ne_iff


/-- An additive character is *nontrivial* if it takes a value `≠ 1`. -/
@[deprecated "No deprecation message was provided." (since := "2024-06-06")]
def IsNontrivial (ψ : AddChar A M) : Prop := ∃ a : A, ψ a ≠ 1


set_option linter.deprecated false in
/-- An additive character is nontrivial iff it is not the trivial character. -/
@[deprecated ne_one_iff (since := "2024-06-06")]
lemma isNontrivial_iff_ne_trivial (ψ : AddChar A M) : IsNontrivial ψ ↔ ψ ≠ 1 :=
  not_forall.symm.trans (DFunLike.ext_iff (f := ψ) (g := 1)).symm.not


noncomputable instance : DecidableEq (AddChar A M) := Classical.decEq _


/-- When `M` is commutative, `AddChar A M` is a commutative monoid. -/
instance instCommMonoid : CommMonoid (AddChar A M) := toMonoidHomEquiv.commMonoid

/-- When `M` is commutative, `AddChar A M` is an additive commutative monoid. -/
instance instAddCommMonoid : AddCommMonoid (AddChar A M) := Additive.addCommMonoid


@[simp, norm_cast] lemma coe_mul (ψ χ : AddChar A M) : ⇑(ψ * χ) = ψ * χ := rfl

@[simp, norm_cast] lemma coe_add (ψ χ : AddChar A M) : ⇑(ψ + χ) = ψ * χ := rfl

@[simp, norm_cast] lemma coe_pow (ψ : AddChar A M) (n : ℕ) : ⇑(ψ ^ n) = ψ ^ n := rfl

@[simp, norm_cast] lemma coe_nsmul (n : ℕ) (ψ : AddChar A M) : ⇑(n • ψ) = ψ ^ n := rfl


@[simp, norm_cast]
lemma coe_prod (s : Finset ι) (ψ : ι → AddChar A M) : ∏ i in s, ψ i = ∏ i in s, ⇑(ψ i) := by
  /-
    ι : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝¹ : AddMonoid A
    inst✝ : CommMonoid M
    s : Finset ι
    ψ : ι → AddChar A M
    ⊢ Eq (⇑(s.prod fun i => ψ i)) (s.prod fun i => ⇑(ψ i))
  -/
                                              /-
                                                🎉 no goals
                                              -/
  induction s using Finset.cons_induction <;> simp [*]
                                              /-
                                                🎉 no goals
                                              -/


@[simp, norm_cast]
lemma coe_sum (s : Finset ι) (ψ : ι → AddChar A M) : ∑ i in s, ψ i = ∏ i in s, ⇑(ψ i) := by
  /-
    ι : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝¹ : AddMonoid A
    inst✝ : CommMonoid M
    s : Finset ι
    ψ : ι → AddChar A M
    ⊢ Eq (⇑(s.sum fun i => ψ i)) (s.prod fun i => ⇑(ψ i))
  -/
                                              /-
                                                🎉 no goals
                                              -/
  induction s using Finset.cons_induction <;> simp [*]
                                              /-
                                                🎉 no goals
                                              -/


@[simp] lemma mul_apply (ψ φ : AddChar A M) (a : A) : (ψ * φ) a = ψ a * φ a := rfl

@[simp] lemma add_apply (ψ φ : AddChar A M) (a : A) : (ψ + φ) a = ψ a * φ a := rfl

@[simp] lemma pow_apply (ψ : AddChar A M) (n : ℕ) (a : A) : (ψ ^ n) a = (ψ a) ^ n := rfl

@[simp] lemma nsmul_apply (ψ : AddChar A M) (n : ℕ) (a : A) : (n • ψ) a = (ψ a) ^ n := rfl


lemma prod_apply (s : Finset ι) (ψ : ι → AddChar A M) (a : A) :
                                              /-
                                                ι : Type u_1
                                                A : Type u_2
                                                M : Type u_3
                                                inst✝¹ : AddMonoid A
                                                inst✝ : CommMonoid M
                                                s : Finset ι
                                                ψ : ι → AddChar A M
                                                a : A
                                                ⊢ Eq ((s.prod fun i => ψ i) a) (s.prod fun i => (ψ i) a)
                                              -/
    (∏ i in s, ψ i) a = ∏ i in s, ψ i a := by rw [coe_prod, Finset.prod_apply]
                                              /-
                                                🎉 no goals
                                              -/


lemma sum_apply (s : Finset ι) (ψ : ι → AddChar A M) (a : A) :
                                              /-
                                                ι : Type u_1
                                                A : Type u_2
                                                M : Type u_3
                                                inst✝¹ : AddMonoid A
                                                inst✝ : CommMonoid M
                                                s : Finset ι
                                                ψ : ι → AddChar A M
                                                a : A
                                                ⊢ Eq ((s.sum fun i => ψ i) a) (s.prod fun i => (ψ i) a)
                                              -/
    (∑ i in s, ψ i) a = ∏ i in s, ψ i a := by rw [coe_sum, Finset.prod_apply]
                                              /-
                                                🎉 no goals
                                              -/


lemma mul_eq_add (ψ χ : AddChar A M) : ψ * χ = ψ + χ := rfl

lemma pow_eq_nsmul (ψ : AddChar A M) (n : ℕ) : ψ ^ n = n • ψ := rfl

lemma prod_eq_sum (s : Finset ι) (ψ : ι → AddChar A M) : ∏ i in s, ψ i = ∑ i in s, ψ i := rfl


@[simp] lemma toMonoidHomEquiv_add (ψ φ : AddChar A M) :
    toMonoidHomEquiv (ψ + φ) = toMonoidHomEquiv ψ * toMonoidHomEquiv φ := rfl

@[simp] lemma toMonoidHomEquiv_symm_mul (ψ φ : Multiplicative A →* M) :
    toMonoidHomEquiv.symm (ψ * φ) = toMonoidHomEquiv.symm ψ + toMonoidHomEquiv.symm φ := rfl


/-- The natural equivalence to `(Multiplicative A →* M)` is a monoid isomorphism. -/
def toMonoidHomMulEquiv : AddChar A M ≃* (Multiplicative A →* M) :=
                                                   /-
                                                     ι : Type u_1
                                                     A : Type u_2
                                                     M : Type u_3
                                                     inst✝¹ : AddMonoid A
                                                     inst✝ : CommMonoid M
                                                     φ ψ : AddChar A M
                                                     ⊢ Eq (__src✝.toFun (HMul.hMul φ ψ)) (HMul.hMul (__src✝.toFun φ) (__src✝.toFun  …
                                                   -/
  { toMonoidHomEquiv with map_mul' := fun φ ψ ↦ by rfl }
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- Additive characters `A → M` are the same thing as additive homomorphisms from `A` to
`Additive M`. -/
def toAddMonoidAddEquiv : Additive (AddChar A M) ≃+ (A →+ Additive M) :=
                                                      /-
                                                        ι : Type u_1
                                                        A : Type u_2
                                                        M : Type u_3
                                                        inst✝¹ : AddMonoid A
                                                        inst✝ : CommMonoid M
                                                        φ ψ : Additive (AddChar A M)
                                                        ⊢ Eq (__src✝.toFun (HAdd.hAdd φ ψ)) (HAdd.hAdd (__src✝.toFun φ) (__src✝.toFun  …
                                                      -/
  { toAddMonoidHomEquiv with map_add' := fun φ ψ ↦ by rfl }
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- The double dual embedding. -/
def doubleDualEmb : A →+ AddChar (AddChar A M) M where
  toFun a := { toFun := fun ψ ↦ ψ a
                                      /-
                                        ι : Type u_1
                                        A : Type u_2
                                        M : Type u_3
                                        inst✝¹ : AddMonoid A
                                        inst✝ : CommMonoid M
                                        a : A
                                        ⊢ Eq ((fun ψ => ψ a) 0) 1
                                      -/
               map_zero_eq_one' := by simp
                                      /-
                                        🎉 no goals
                                      -/
                                     /-
                                       ι : Type u_1
                                       A : Type u_2
                                       M : Type u_3
                                       inst✝¹ : AddMonoid A
                                       inst✝ : CommMonoid M
                                       a : A
                                       ⊢ ∀ (a_1 b : AddChar A M), Eq ((fun ψ => ψ a) (HAdd.hAdd a_1 b)) (HMul.hMul (( …
                                     -/
               map_add_eq_mul' := by simp }
                                     /-
                                       🎉 no goals
                                     -/
                  /-
                    ι : Type u_1
                    A : Type u_2
                    M : Type u_3
                    inst✝¹ : AddMonoid A
                    inst✝ : CommMonoid M
                    ⊢ Eq ((fun a => { toFun := fun ψ => ψ a, map_zero_eq_one' := ⋯, map_add_eq_mul …
                  -/
  map_zero' := by ext; simp
                       /-
                         🎉 no goals
                       -/
                     /-
                       ι : Type u_1
                       A : Type u_2
                       M : Type u_3
                       inst✝¹ : AddMonoid A
                       inst✝ : CommMonoid M
                       x✝¹ x✝ : A
                       ⊢ Eq ({ toFun := fun a => { toFun := fun ψ => ψ a, map_zero_eq_one' := ⋯, map_ …
                     -/
  map_add' _ _ := by ext; simp [map_add_eq_mul]
                          /-
                            🎉 no goals
                          -/


@[simp] lemma doubleDualEmb_apply (a : A) (ψ : AddChar A M) : doubleDualEmb a ψ = ψ a := rfl


lemma sum_eq_ite (ψ : AddChar A R) [Decidable (ψ = 0)] :
    ∑ a, ψ a = if ψ = 0 then ↑(card A) else 0 := by
  /-
    A : Type u_1
    R : Type u_2
    inst✝⁴ : AddGroup A
    inst✝³ : Fintype A
    inst✝² : CommSemiring R
    inst✝¹ : IsDomain R
    ψ : AddChar A R
    inst✝ : Decidable (Eq ψ 0)
    ⊢ Eq (Finset.univ.sum fun a => ψ a) (ite (Eq ψ 0) (↑(Fintype.card A)) 0)
  -/
  split_ifs with h
    /-
      case pos
      A : Type u_1
      R : Type u_2
      inst✝⁴ : AddGroup A
      inst✝³ : Fintype A
      inst✝² : CommSemiring R
      inst✝¹ : IsDomain R
      ψ : AddChar A R
      inst✝ : Decidable (Eq ψ 0)
      h : Eq ψ 0
      ⊢ Eq (Finset.univ.sum fun a => ψ a) ↑(Fintype.card A)
    -/
  · simp [h, card_univ]
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_1
    R : Type u_2
    inst✝⁴ : AddGroup A
    inst✝³ : Fintype A
    inst✝² : CommSemiring R
    inst✝¹ : IsDomain R
    ψ : AddChar A R
    inst✝ : Decidable (Eq ψ 0)
    h : Not (Eq ψ 0)
    ⊢ Eq (Finset.univ.sum fun a => ψ a) 0
  -/
  obtain ⟨x, hx⟩ := ne_one_iff.1 h
  /-
    case neg.intro
    A : Type u_1
    R : Type u_2
    inst✝⁴ : AddGroup A
    inst✝³ : Fintype A
    inst✝² : CommSemiring R
    inst✝¹ : IsDomain R
    ψ : AddChar A R
    inst✝ : Decidable (Eq ψ 0)
    h : Not (Eq ψ 0)
    x : A
    hx : Ne (ψ x) 1
    ⊢ Eq (Finset.univ.sum fun a => ψ a) 0
  -/
  refine eq_zero_of_mul_eq_self_left hx ?_
  /-
    case neg.intro
    A : Type u_1
    R : Type u_2
    inst✝⁴ : AddGroup A
    inst✝³ : Fintype A
    inst✝² : CommSemiring R
    inst✝¹ : IsDomain R
    ψ : AddChar A R
    inst✝ : Decidable (Eq ψ 0)
    h : Not (Eq ψ 0)
    x : A
    hx : Ne (ψ x) 1
    ⊢ Eq (HMul.hMul (ψ x) (Finset.univ.sum fun a => ψ a)) (Finset.univ.sum fun a = …
  -/
  rw [Finset.mul_sum]
  /-
    case neg.intro
    A : Type u_1
    R : Type u_2
    inst✝⁴ : AddGroup A
    inst✝³ : Fintype A
    inst✝² : CommSemiring R
    inst✝¹ : IsDomain R
    ψ : AddChar A R
    inst✝ : Decidable (Eq ψ 0)
    h : Not (Eq ψ 0)
    x : A
    hx : Ne (ψ x) 1
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (ψ x) (ψ i)) (Finset.univ.sum fun a = …
  -/
  exact Fintype.sum_equiv (Equiv.addLeft x) _ _ fun y ↦ (map_add_eq_mul ..).symm
  /-
    🎉 no goals
  -/


lemma sum_eq_zero_iff_ne_zero : ∑ x, ψ x = 0 ↔ ψ ≠ 0 := by
  classical
  rw [sum_eq_ite, Ne.ite_eq_right_iff]; exact Nat.cast_ne_zero.2 Fintype.card_ne_zero


lemma sum_ne_zero_iff_eq_zero : ∑ x, ψ x ≠ 0 ↔ ψ = 0 := sum_eq_zero_iff_ne_zero.not_left


/-- The additive characters on a commutative additive group form a commutative group.

Note that the inverse is defined using negation on the domain; we do not assume `M` has an
inversion operation for the definition (but see `AddChar.map_neg_eq_inv` below). -/
instance instCommGroup : CommGroup (AddChar A M) :=
  { instCommMonoid with
    inv := fun ψ ↦ ψ.compAddMonoidHom negAddMonoidHom
                                 /-
                                   A : Type u_1
                                   M : Type u_2
                                   inst✝¹ : AddCommGroup A
                                   inst✝ : CommMonoid M
                                   ψ : AddChar A M
                                   ⊢ Eq (HMul.hMul (Inv.inv ψ) ψ) 1
                                 -/
    inv_mul_cancel := fun ψ ↦ by ext1 x; simp [negAddMonoidHom, ← map_add_eq_mul]}
                                         /-
                                           🎉 no goals
                                         -/


/-- The additive characters on a commutative additive group form a commutative group. -/
instance : AddCommGroup (AddChar A M) := Additive.addCommGroup


@[simp] lemma inv_apply (ψ : AddChar A M) (a : A) : ψ⁻¹ a = ψ (-a) := rfl

@[simp] lemma neg_apply (ψ : AddChar A M) (a : A) : (-ψ) a = ψ (-a) := rfl

lemma div_apply (ψ χ : AddChar A M) (a : A) : (ψ / χ) a = ψ a * χ (-a) := rfl

lemma sub_apply (ψ χ : AddChar A M) (a : A) : (ψ - χ) a = ψ a * χ (-a) := rfl


/-- The values of an additive character on an additive group are units. -/
lemma val_isUnit {A M} [AddGroup A] [Monoid M] (φ : AddChar A M) (a : A) : IsUnit (φ a) :=
  IsUnit.map φ.toMonoidHom <| Group.isUnit (Multiplicative.ofAdd a)


/-- An additive character maps negatives to inverses (when defined) -/
lemma map_neg_eq_inv (ψ : AddChar A M) (a : A) : ψ (-a) = (ψ a)⁻¹ := by
  /-
    A : Type u_1
    M : Type u_2
    inst✝¹ : AddGroup A
    inst✝ : DivisionMonoid M
    ψ : AddChar A M
    a : A
    ⊢ Eq (ψ (Neg.neg a)) (Inv.inv (ψ a))
  -/
  apply eq_inv_of_mul_eq_one_left
  /-
    case h
    A : Type u_1
    M : Type u_2
    inst✝¹ : AddGroup A
    inst✝ : DivisionMonoid M
    ψ : AddChar A M
    a : A
    ⊢ Eq (HMul.hMul (ψ (Neg.neg a)) (ψ a)) 1
  -/
  simp only [← map_add_eq_mul, neg_add_cancel, map_zero_eq_one]
  /-
    🎉 no goals
  -/


/-- An additive character maps integer scalar multiples to integer powers. -/
lemma map_zsmul_eq_zpow (ψ : AddChar A M) (n : ℤ) (a : A) : ψ (n • a) = (ψ a) ^ n :=
  ψ.toMonoidHom.map_zpow a n


@[deprecated (since := "2024-06-06")] alias map_neg_inv := map_neg_eq_inv

@[deprecated (since := "2024-06-06")] alias map_zsmul_zpow := map_zsmul_eq_zpow


                                                                   /-
                                                                     A : Type u_1
                                                                     M : Type u_2
                                                                     inst✝¹ : AddCommGroup A
                                                                     inst✝ : DivisionCommMonoid M
                                                                     ψ : AddChar A M
                                                                     a : A
                                                                     ⊢ Eq ((Inv.inv ψ) a) (Inv.inv (ψ a))
                                                                   -/
lemma inv_apply' (ψ : AddChar A M) (a : A) : ψ⁻¹ a = (ψ a)⁻¹ := by rw [inv_apply, map_neg_eq_inv]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

lemma neg_apply' (ψ : AddChar A M) (a : A) : (-ψ) a = (ψ a)⁻¹ := map_neg_eq_inv _ _


lemma div_apply' (ψ χ : AddChar A M) (a : A) : (ψ / χ) a = ψ a / χ a := by
  /-
    A : Type u_1
    M : Type u_2
    inst✝¹ : AddCommGroup A
    inst✝ : DivisionCommMonoid M
    ψ χ : AddChar A M
    a : A
    ⊢ Eq ((HDiv.hDiv ψ χ) a) (HDiv.hDiv (ψ a) (χ a))
  -/
  rw [div_apply, map_neg_eq_inv, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


lemma sub_apply' (ψ χ : AddChar A M) (a : A) : (ψ - χ) a = ψ a / χ a := by
  /-
    A : Type u_1
    M : Type u_2
    inst✝¹ : AddCommGroup A
    inst✝ : DivisionCommMonoid M
    ψ χ : AddChar A M
    a : A
    ⊢ Eq ((HSub.hSub ψ χ) a) (HDiv.hDiv (ψ a) (χ a))
  -/
  rw [sub_apply, map_neg_eq_inv, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[simp] lemma zsmul_apply (n : ℤ) (ψ : AddChar A M) (a : A) : (n • ψ) a = ψ a ^ n := by
  /-
    A : Type u_1
    M : Type u_2
    inst✝¹ : AddCommGroup A
    inst✝ : DivisionCommMonoid M
    n : Int
    ψ : AddChar A M
    a : A
    ⊢ Eq ((HSMul.hSMul n ψ) a) (HPow.hPow (ψ a) n)
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp [-neg_apply, neg_apply']
              /-
                🎉 no goals
              -/


@[simp] lemma zpow_apply (ψ : AddChar A M) (n : ℤ) (a : A) : (ψ ^ n) a = ψ a ^ n := zsmul_apply ..


lemma map_sub_eq_div (ψ : AddChar A M) (a b : A) : ψ (a - b) = ψ a / ψ b :=
  ψ.toMonoidHom.map_div _ _


lemma injective_iff {ψ : AddChar A M} : Injective ψ ↔ ∀ ⦃x⦄, ψ x = 1 → x = 0 :=
  ψ.toMonoidHom.ker_eq_bot_iff.symm.trans eq_bot_iff


@[simp] lemma coe_ne_zero (ψ : AddChar A M₀) : (ψ : A → M₀) ≠ 0 :=
                          /-
                            A : Type u_1
                            M₀ : Type u_2
                            inst✝² : AddGroup A
                            inst✝¹ : MonoidWithZero M₀
                            inst✝ : Nontrivial M₀
                            ψ : AddChar A M₀
                            h : Eq (ψ 0) (0 0)
                            ⊢ False
                          -/
  ne_iff.2 ⟨0, fun h ↦ by simpa only [h, Pi.zero_apply, zero_ne_one] using map_zero_eq_one ψ⟩
                          /-
                            🎉 no goals
                          -/


/-- Define the multiplicative shift of an additive character.
This satisfies `mulShift ψ a x = ψ (a * x)`. -/
def mulShift (ψ : AddChar R M) (r : R) : AddChar R M :=
  ψ.compAddMonoidHom (AddMonoidHom.mulLeft r)


@[simp] lemma mulShift_apply {ψ : AddChar R M} {r : R} {x : R} : mulShift ψ r x = ψ (r * x) :=
  rfl


/-- `ψ⁻¹ = mulShift ψ (-1))`. -/
theorem inv_mulShift (ψ : AddChar R M) : ψ⁻¹ = mulShift ψ (-1) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    ⊢ Eq (Inv.inv ψ) (ψ.mulShift (-1))
  -/
  ext
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    x✝ : R
    ⊢ Eq ((Inv.inv ψ) x✝) ((ψ.mulShift (-1)) x✝)
  -/
  rw [inv_apply, mulShift_apply, neg_mul, one_mul]
  /-
    🎉 no goals
  -/


/-- If `n` is a natural number, then `mulShift ψ n x = (ψ x) ^ n`. -/
theorem mulShift_spec' (ψ : AddChar R M) (n : ℕ) (x : R) : mulShift ψ n x = ψ x ^ n := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    n : Nat
    x : R
    ⊢ Eq ((ψ.mulShift ↑n) x) (HPow.hPow (ψ x) n)
  -/
  rw [mulShift_apply, ← nsmul_eq_mul, map_nsmul_eq_pow]
  /-
    🎉 no goals
  -/


/-- If `n` is a natural number, then `ψ ^ n = mulShift ψ n`. -/
theorem pow_mulShift (ψ : AddChar R M) (n : ℕ) : ψ ^ n = mulShift ψ n := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    n : Nat
    ⊢ Eq (HPow.hPow ψ n) (ψ.mulShift ↑n)
  -/
  ext x
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    n : Nat
    x : R
    ⊢ Eq ((HPow.hPow ψ n) x) ((ψ.mulShift ↑n) x)
  -/
  rw [pow_apply, ← mulShift_spec']
  /-
    🎉 no goals
  -/


/-- The product of `mulShift ψ r` and `mulShift ψ s` is `mulShift ψ (r + s)`. -/
theorem mulShift_mul (ψ : AddChar R M) (r s : R) :
    mulShift ψ r * mulShift ψ s = mulShift ψ (r + s) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    r s : R
    ⊢ Eq (HMul.hMul (ψ.mulShift r) (ψ.mulShift s)) (ψ.mulShift (HAdd.hAdd r s))
  -/
  ext
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    r s x✝ : R
    ⊢ Eq ((HMul.hMul (ψ.mulShift r) (ψ.mulShift s)) x✝) ((ψ.mulShift (HAdd.hAdd r  …
  -/
  rw [mulShift_apply, right_distrib, map_add_eq_mul]; norm_cast
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma mulShift_mulShift (ψ : AddChar R M) (r s : R) :
    mulShift (mulShift ψ r) s = mulShift ψ (r * s) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    r s : R
    ⊢ Eq ((ψ.mulShift r).mulShift s) (ψ.mulShift (HMul.hMul r s))
  -/
  ext
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    r s x✝ : R
    ⊢ Eq (((ψ.mulShift r).mulShift s) x✝) ((ψ.mulShift (HMul.hMul r s)) x✝)
  -/
  simp only [mulShift_apply, mul_assoc]
  /-
    🎉 no goals
  -/


/-- `mulShift ψ 0` is the trivial character. -/
@[simp]
theorem mulShift_zero (ψ : AddChar R M) : mulShift ψ 0 = 1 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    ⊢ Eq (ψ.mulShift 0) 1
  -/
  ext; rw [mulShift_apply, zero_mul, map_zero_eq_one, one_apply]
       /-
         🎉 no goals
       -/


@[simp]
lemma mulShift_one (ψ : AddChar R M) : mulShift ψ 1 = ψ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    ⊢ Eq (ψ.mulShift 1) ψ
  -/
  ext; rw [mulShift_apply, one_mul]
       /-
         🎉 no goals
       -/


lemma mulShift_unit_eq_one_iff (ψ : AddChar R M) {u : R} (hu : IsUnit u) :
    ψ.mulShift u = 1 ↔ ψ = 1 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : Ring R
    inst✝ : CommMonoid M
    ψ : AddChar R M
    u : R
    hu : IsUnit u
    ⊢ Iff (Eq (ψ.mulShift u) 1) (Eq ψ 1)
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝¹ : Ring R
      inst✝ : CommMonoid M
      ψ : AddChar R M
      u : R
      hu : IsUnit u
      h : Eq (ψ.mulShift u) 1
      ⊢ Eq ψ 1
    -/
  · ext1 y
    /-
      case refine_1.h
      R : Type u_1
      M : Type u_2
      inst✝¹ : Ring R
      inst✝ : CommMonoid M
      ψ : AddChar R M
      u : R
      hu : IsUnit u
      h : Eq (ψ.mulShift u) 1
      y : R
      ⊢ Eq (ψ y) (1 y)
    -/
    rw [show y = u * (hu.unit⁻¹ * y) by rw [← mul_assoc, IsUnit.mul_val_inv, one_mul]]
    /-
      case refine_1.h
      R : Type u_1
      M : Type u_2
      inst✝¹ : Ring R
      inst✝ : CommMonoid M
      ψ : AddChar R M
      u : R
      hu : IsUnit u
      h : Eq (ψ.mulShift u) 1
      y : R
      ⊢ Eq (ψ (HMul.hMul u (HMul.hMul (↑(Inv.inv hu.unit)) y))) (1 (HMul.hMul u (HMu …
    -/
    simpa only [mulShift_apply] using DFunLike.ext_iff.mp h (hu.unit⁻¹ * y)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝¹ : Ring R
      inst✝ : CommMonoid M
      ψ : AddChar R M
      u : R
      hu : IsUnit u
      ⊢ Eq ψ 1 → Eq (ψ.mulShift u) 1
    -/
  · rintro rfl
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝¹ : Ring R
      inst✝ : CommMonoid M
      u : R
      hu : IsUnit u
      ⊢ Eq (AddChar.mulShift 1 u) 1
    -/
    ext1 y
    /-
      case refine_2.h
      R : Type u_1
      M : Type u_2
      inst✝¹ : Ring R
      inst✝ : CommMonoid M
      u : R
      hu : IsUnit u
      y : R
      ⊢ Eq ((AddChar.mulShift 1 u) y) (1 y)
    -/
    rw [mulShift_apply, one_apply, one_apply]
    /-
      🎉 no goals
    -/


