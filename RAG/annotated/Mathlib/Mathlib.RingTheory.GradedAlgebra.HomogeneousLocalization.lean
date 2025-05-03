local notation "at " x => Localization x


/-- Let `x` be a submonoid of `A`, then `NumDenSameDeg 𝒜 x` is a structure with a numerator and a
denominator with same grading such that the denominator is contained in `x`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure NumDenSameDeg where
  deg : ι
  (num den : 𝒜 deg)
  den_mem : (den : A) ∈ x


@[ext]
theorem ext {c1 c2 : NumDenSameDeg 𝒜 x} (hdeg : c1.deg = c2.deg) (hnum : (c1.num : A) = c2.num)
    (hden : (c1.den : A) = c2.den) : c1 = c2 := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
    hdeg : Eq c1.deg c2.deg
    hnum : Eq ↑c1.num ↑c2.num
    hden : Eq ↑c1.den ↑c2.den
    ⊢ Eq c1 c2
  -/
  rcases c1 with ⟨i1, ⟨n1, hn1⟩, ⟨d1, hd1⟩, h1⟩
  /-
    case mk.mk.mk
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
    i1 : ι
    n1 : A
    hn1 : Membership.mem (𝒜 i1) n1
    d1 : A
    hd1 : Membership.mem (𝒜 i1) d1
    h1 : Membership.mem x ↑⟨d1, hd1⟩
    hdeg : Eq { deg := i1, num := ⟨n1, hn1⟩, den := ⟨d1, hd1⟩, den_mem := h1 }.deg …
    hnum : Eq ↑{ deg := i1, num := ⟨n1, hn1⟩, den := ⟨d1, hd1⟩, den_mem := h1 }.nu …
    hden : Eq ↑{ deg := i1, num := ⟨n1, hn1⟩, den := ⟨d1, hd1⟩, den_mem := h1 }.de …
    ⊢ Eq { deg := i1, num := ⟨n1, hn1⟩, den := ⟨d1, hd1⟩, den_mem := h1 } c2
  -/
  rcases c2 with ⟨i2, ⟨n2, hn2⟩, ⟨d2, hd2⟩, h2⟩
  /-
    case mk.mk.mk.mk.mk.mk
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    i1 : ι
    n1 : A
    hn1 : Membership.mem (𝒜 i1) n1
    d1 : A
    hd1 : Membership.mem (𝒜 i1) d1
    h1 : Membership.mem x ↑⟨d1, hd1⟩
    i2 : ι
    n2 : A
    hn2 : Membership.mem (𝒜 i2) n2
    d2 : A
    hd2 : Membership.mem (𝒜 i2) d2
    h2 : Membership.mem x ↑⟨d2, hd2⟩
    hdeg : Eq { deg := i1, num := ⟨n1, hn1⟩, den := ⟨d1, hd1⟩, den_mem := h1 }.deg …
    hnum : Eq ↑{ deg := i1, num := ⟨n1, hn1⟩, den := ⟨d1, hd1⟩, den_mem := h1 }.nu …
    hden : Eq ↑{ deg := i1, num := ⟨n1, hn1⟩, den := ⟨d1, hd1⟩, den_mem := h1 }.de …
    ⊢ Eq { deg := i1, num := ⟨n1, hn1⟩, den := ⟨d1, hd1⟩, den_mem := h1 } { deg := …
  -/
  dsimp only [Subtype.coe_mk] at *
  /-
    case mk.mk.mk.mk.mk.mk
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    i1 : ι
    n1 : A
    hn1 : Membership.mem (𝒜 i1) n1
    d1 : A
    hd1 : Membership.mem (𝒜 i1) d1
    h1 : Membership.mem x ↑⟨d1, hd1⟩
    i2 : ι
    n2 : A
    hn2 : Membership.mem (𝒜 i2) n2
    d2 : A
    hd2 : Membership.mem (𝒜 i2) d2
    h2 : Membership.mem x ↑⟨d2, hd2⟩
    hdeg : Eq i1 i2
    hnum : Eq n1 n2
    hden : Eq d1 d2
    ⊢ Eq { deg := i1, num := ⟨n1, hn1⟩, den := ⟨d1, hd1⟩, den_mem := h1 } { deg := …
  -/
  subst hdeg hnum hden
  /-
    case mk.mk.mk.mk.mk.mk
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    i1 : ι
    n1 : A
    hn1 : Membership.mem (𝒜 i1) n1
    d1 : A
    hd1 : Membership.mem (𝒜 i1) d1
    h1 : Membership.mem x ↑⟨d1, hd1⟩
    hn2 : Membership.mem (𝒜 i1) n1
    hd2 : Membership.mem (𝒜 i1) d1
    h2 : Membership.mem x ↑⟨d1, hd2⟩
    ⊢ Eq { deg := i1, num := ⟨n1, hn1⟩, den := ⟨d1, hd1⟩, den_mem := h1 } { deg := …
  -/
  congr
  /-
    🎉 no goals
  -/


instance : Neg (NumDenSameDeg 𝒜 x) where
  neg c := ⟨c.deg, ⟨-c.num, neg_mem c.num.2⟩, c.den, c.den_mem⟩


@[simp]
theorem deg_neg (c : NumDenSameDeg 𝒜 x) : (-c).deg = c.deg :=
  rfl


@[simp]
theorem num_neg (c : NumDenSameDeg 𝒜 x) : ((-c).num : A) = -c.num :=
  rfl


@[simp]
theorem den_neg (c : NumDenSameDeg 𝒜 x) : ((-c).den : A) = c.den :=
  rfl


instance : SMul α (NumDenSameDeg 𝒜 x) where
  smul m c := ⟨c.deg, m • c.num, c.den, c.den_mem⟩


@[simp]
theorem deg_smul (c : NumDenSameDeg 𝒜 x) (m : α) : (m • c).deg = c.deg :=
  rfl


@[simp]
theorem num_smul (c : NumDenSameDeg 𝒜 x) (m : α) : ((m • c).num : A) = m • c.num :=
  rfl


@[simp]
theorem den_smul (c : NumDenSameDeg 𝒜 x) (m : α) : ((m • c).den : A) = c.den :=
  rfl


instance : One (NumDenSameDeg 𝒜 x) where
  one :=
    { deg := 0
      -- Porting note: Changed `one_mem` to `GradedOne.one_mem`
      num := ⟨1, GradedOne.one_mem⟩
      den := ⟨1, GradedOne.one_mem⟩
      den_mem := Submonoid.one_mem _ }


@[simp]
theorem deg_one : (1 : NumDenSameDeg 𝒜 x).deg = 0 :=
  rfl


@[simp]
theorem num_one : ((1 : NumDenSameDeg 𝒜 x).num : A) = 1 :=
  rfl


@[simp]
theorem den_one : ((1 : NumDenSameDeg 𝒜 x).den : A) = 1 :=
  rfl


instance : Zero (NumDenSameDeg 𝒜 x) where
  zero := ⟨0, 0, ⟨1, GradedOne.one_mem⟩, Submonoid.one_mem _⟩


@[simp]
theorem deg_zero : (0 : NumDenSameDeg 𝒜 x).deg = 0 :=
  rfl


@[simp]
theorem num_zero : (0 : NumDenSameDeg 𝒜 x).num = 0 :=
  rfl


@[simp]
theorem den_zero : ((0 : NumDenSameDeg 𝒜 x).den : A) = 1 :=
  rfl


instance : Mul (NumDenSameDeg 𝒜 x) where
  mul p q :=
    { deg := p.deg + q.deg
      -- Porting note: Changed `mul_mem` to `GradedMul.mul_mem`
      num := ⟨p.num * q.num, GradedMul.mul_mem p.num.prop q.num.prop⟩
      den := ⟨p.den * q.den, GradedMul.mul_mem p.den.prop q.den.prop⟩
      den_mem := Submonoid.mul_mem _ p.den_mem q.den_mem }


@[simp]
theorem deg_mul (c1 c2 : NumDenSameDeg 𝒜 x) : (c1 * c2).deg = c1.deg + c2.deg :=
  rfl


@[simp]
theorem num_mul (c1 c2 : NumDenSameDeg 𝒜 x) : ((c1 * c2).num : A) = c1.num * c2.num :=
  rfl


@[simp]
theorem den_mul (c1 c2 : NumDenSameDeg 𝒜 x) : ((c1 * c2).den : A) = c1.den * c2.den :=
  rfl


instance : Add (NumDenSameDeg 𝒜 x) where
  add c1 c2 :=
    { deg := c1.deg + c2.deg
      num := ⟨c1.den * c2.num + c2.den * c1.num,
        add_mem (GradedMul.mul_mem c1.den.2 c2.num.2)
          (add_comm c2.deg c1.deg ▸ GradedMul.mul_mem c2.den.2 c1.num.2)⟩
      den := ⟨c1.den * c2.den, GradedMul.mul_mem c1.den.2 c2.den.2⟩
      den_mem := Submonoid.mul_mem _ c1.den_mem c2.den_mem }


@[simp]
theorem deg_add (c1 c2 : NumDenSameDeg 𝒜 x) : (c1 + c2).deg = c1.deg + c2.deg :=
  rfl


@[simp]
theorem num_add (c1 c2 : NumDenSameDeg 𝒜 x) :
    ((c1 + c2).num : A) = c1.den * c2.num + c2.den * c1.num :=
  rfl


@[simp]
theorem den_add (c1 c2 : NumDenSameDeg 𝒜 x) : ((c1 + c2).den : A) = c1.den * c2.den :=
  rfl


instance : CommMonoid (NumDenSameDeg 𝒜 x) where
  one := 1
  mul := (· * ·)
  mul_assoc _ _ _ := ext _ (add_assoc _ _ _) (mul_assoc _ _ _) (mul_assoc _ _ _)
  one_mul _ := ext _ (zero_add _) (one_mul _) (one_mul _)
  mul_one _ := ext _ (add_zero _) (mul_one _) (mul_one _)
  mul_comm _ _ := ext _ (add_comm _ _) (mul_comm _ _) (mul_comm _ _)


instance : Pow (NumDenSameDeg 𝒜 x) ℕ where
  pow c n :=
    ⟨n • c.deg, @GradedMonoid.GMonoid.gnpow _ (fun i => ↥(𝒜 i)) _ _ n _ c.num,
      @GradedMonoid.GMonoid.gnpow _ (fun i => ↥(𝒜 i)) _ _ n _ c.den, by
        /-
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          inst✝⁵ : CommRing R
          inst✝⁴ : CommRing A
          inst✝³ : Algebra R A
          𝒜 : ι → Submodule R A
          x : Submonoid A
          inst✝² : AddCommMonoid ι
          inst✝¹ : DecidableEq ι
          inst✝ : GradedAlgebra 𝒜
          c : HomogeneousLocalization.NumDenSameDeg 𝒜 x
          n : Nat
          ⊢ Membership.mem x ↑(GradedMonoid.GMonoid.gnpow n c.den)
        -/
        induction' n with n ih
          /-
            case zero
            ι : Type u_1
            R : Type u_2
            A : Type u_3
            inst✝⁵ : CommRing R
            inst✝⁴ : CommRing A
            inst✝³ : Algebra R A
            𝒜 : ι → Submodule R A
            x : Submonoid A
            inst✝² : AddCommMonoid ι
            inst✝¹ : DecidableEq ι
            inst✝ : GradedAlgebra 𝒜
            c : HomogeneousLocalization.NumDenSameDeg 𝒜 x
            ⊢ Membership.mem x ↑(GradedMonoid.GMonoid.gnpow 0 c.den)
          -/
        · simpa only [coe_gnpow, pow_zero] using Submonoid.one_mem _
          /-
            🎉 no goals
          -/
          /-
            case succ
            ι : Type u_1
            R : Type u_2
            A : Type u_3
            inst✝⁵ : CommRing R
            inst✝⁴ : CommRing A
            inst✝³ : Algebra R A
            𝒜 : ι → Submodule R A
            x : Submonoid A
            inst✝² : AddCommMonoid ι
            inst✝¹ : DecidableEq ι
            inst✝ : GradedAlgebra 𝒜
            c : HomogeneousLocalization.NumDenSameDeg 𝒜 x
            n : Nat
            ih : Membership.mem x ↑(GradedMonoid.GMonoid.gnpow n c.den)
            ⊢ Membership.mem x ↑(GradedMonoid.GMonoid.gnpow (HAdd.hAdd n 1) c.den)
          -/
        · simpa only [pow_succ, coe_gnpow] using x.mul_mem ih c.den_mem⟩
          /-
            🎉 no goals
          -/


@[simp]
theorem deg_pow (c : NumDenSameDeg 𝒜 x) (n : ℕ) : (c ^ n).deg = n • c.deg :=
  rfl


@[simp]
theorem num_pow (c : NumDenSameDeg 𝒜 x) (n : ℕ) : ((c ^ n).num : A) = (c.num : A) ^ n :=
  rfl


@[simp]
theorem den_pow (c : NumDenSameDeg 𝒜 x) (n : ℕ) : ((c ^ n).den : A) = (c.den : A) ^ n :=
  rfl


/-- For `x : prime ideal of A` and any `p : NumDenSameDeg 𝒜 x`, or equivalent a numerator and a
denominator of the same degree, we get an element `p.num / p.den` of `Aₓ`.
-/
def embedding (p : NumDenSameDeg 𝒜 x) : at x :=
  Localization.mk p.num ⟨p.den, p.den_mem⟩


/-- For `x : prime ideal of A`, `HomogeneousLocalization 𝒜 x` is `NumDenSameDeg 𝒜 x` modulo the
kernel of `embedding 𝒜 x`. This is essentially the subring of `Aₓ` where the numerator and
denominator share the same grading.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
def HomogeneousLocalization : Type _ :=
  Quotient (Setoid.ker <| HomogeneousLocalization.NumDenSameDeg.embedding 𝒜 x)


/-- Construct an element of `HomogeneousLocalization 𝒜 x` from a homogeneous fraction. -/
abbrev mk (y : HomogeneousLocalization.NumDenSameDeg 𝒜 x) : HomogeneousLocalization 𝒜 x :=
  Quotient.mk'' y


lemma mk_surjective : Function.Surjective (mk (𝒜 := 𝒜) (x := x)) :=
  Quotient.mk''_surjective


/-- View an element of `HomogeneousLocalization 𝒜 x` as an element of `Aₓ` by forgetting that the
numerator and denominator are of the same grading.
-/
def val (y : HomogeneousLocalization 𝒜 x) : at x :=
  Quotient.liftOn' y (NumDenSameDeg.embedding 𝒜 x) fun _ _ => id


@[simp]
theorem val_mk (i : NumDenSameDeg 𝒜 x) :
    val (mk i) = Localization.mk (i.num : A) ⟨i.den, i.den_mem⟩ :=
  rfl


@[ext]
theorem val_injective : Function.Injective (HomogeneousLocalization.val (𝒜 := 𝒜) (x := x)) :=
  fun a b => Quotient.recOnSubsingleton₂' a b fun _ _ h => Quotient.sound' h


variable (𝒜) {x} in
lemma subsingleton (hx : 0 ∈ x) : Subsingleton (HomogeneousLocalization 𝒜 x) :=
  have := IsLocalization.subsingleton (S := at x) hx
  (HomogeneousLocalization.val_injective (𝒜 := 𝒜) (x := x)).subsingleton


instance : SMul α (HomogeneousLocalization 𝒜 x) where
  smul m := Quotient.map' (m • ·) fun c1 c2 (h : Localization.mk _ _ = Localization.mk _ _) => by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      𝒜 : ι → Submodule R A
      x : Submonoid A
      α : Type u_4
      inst✝³ : SMul α R
      inst✝² : SMul α A
      inst✝¹ : IsScalarTower α R A
      inst✝ : IsScalarTower α A A
      m : α
      c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
      h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
      ⊢ (Setoid.ker (HomogeneousLocalization.NumDenSameDeg.embedding 𝒜 x)) ((fun x_1 …
    -/
    change Localization.mk _ _ = Localization.mk _ _
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      𝒜 : ι → Submodule R A
      x : Submonoid A
      α : Type u_4
      inst✝³ : SMul α R
      inst✝² : SMul α A
      inst✝¹ : IsScalarTower α R A
      inst✝ : IsScalarTower α A A
      m : α
      c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
      h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
      ⊢ Eq (Localization.mk ↑((fun x_1 => HSMul.hSMul m x_1) c1).num ⟨↑((fun x_1 =>  …
    -/
    simp only [num_smul, den_smul]
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      𝒜 : ι → Submodule R A
      x : Submonoid A
      α : Type u_4
      inst✝³ : SMul α R
      inst✝² : SMul α A
      inst✝¹ : IsScalarTower α R A
      inst✝ : IsScalarTower α A A
      m : α
      c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
      h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
      ⊢ Eq (Localization.mk (HSMul.hSMul m ↑c1.num) ⟨↑c1.den, ⋯⟩) (Localization.mk ( …
    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
    convert congr_arg (fun z : at x => m • z) h <;> rw [Localization.smul_mk]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp] lemma mk_smul (i : NumDenSameDeg 𝒜 x) (m : α) : mk (m • i) = m • mk i := rfl


@[simp]
theorem val_smul (n : α) : ∀ y : HomogeneousLocalization 𝒜 x, (n • y).val = n • y.val :=
                           /-
                             ι : Type u_1
                             R : Type u_2
                             A : Type u_3
                             inst✝⁶ : CommRing R
                             inst✝⁵ : CommRing A
                             inst✝⁴ : Algebra R A
                             𝒜 : ι → Submodule R A
                             x : Submonoid A
                             α : Type u_4
                             inst✝³ : SMul α R
                             inst✝² : SMul α A
                             inst✝¹ : IsScalarTower α R A
                             inst✝ : IsScalarTower α A A
                             n : α
                             x✝ : HomogeneousLocalization.NumDenSameDeg 𝒜 x
                             ⊢ Eq (HSMul.hSMul n (Quotient.mk'' x✝)).val (HSMul.hSMul n (HomogeneousLocaliz …
                           -/
  Quotient.ind' fun _ ↦ by rw [← mk_smul, val_mk, val_mk, Localization.smul_mk]; rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem val_nsmul (n : ℕ) (y : HomogeneousLocalization 𝒜 x) : (n • y).val = n • y.val := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    n : Nat
    y : HomogeneousLocalization 𝒜 x
    ⊢ Eq (HSMul.hSMul n y).val (HSMul.hSMul n y.val)
  -/
  rw [val_smul, OreLocalization.nsmul_eq_nsmul]
  /-
    🎉 no goals
  -/


theorem val_zsmul (n : ℤ) (y : HomogeneousLocalization 𝒜 x) : (n • y).val = n • y.val := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    n : Int
    y : HomogeneousLocalization 𝒜 x
    ⊢ Eq (HSMul.hSMul n y).val (HSMul.hSMul n y.val)
  -/
  rw [val_smul, OreLocalization.zsmul_eq_zsmul]
  /-
    🎉 no goals
  -/


instance : Neg (HomogeneousLocalization 𝒜 x) where
  neg := Quotient.map' Neg.neg fun c1 c2 (h : Localization.mk _ _ = Localization.mk _ _) => by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      𝒜 : ι → Submodule R A
      x : Submonoid A
      c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
      h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
      ⊢ (Setoid.ker (HomogeneousLocalization.NumDenSameDeg.embedding 𝒜 x)) (Neg.neg  …
    -/
    change Localization.mk _ _ = Localization.mk _ _
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      𝒜 : ι → Submodule R A
      x : Submonoid A
      c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
      h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
      ⊢ Eq (Localization.mk ↑(Neg.neg c1).num ⟨↑(Neg.neg c1).den, ⋯⟩) (Localization. …
    -/
    simp only [num_neg, den_neg, ← Localization.neg_mk]
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      𝒜 : ι → Submodule R A
      x : Submonoid A
      c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
      h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
      ⊢ Eq (Neg.neg (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩)) (Neg.neg (Localization.m …
    -/
    exact congr_arg Neg.neg h
    /-
      🎉 no goals
    -/


@[simp] lemma mk_neg (i : NumDenSameDeg 𝒜 x) : mk (-i) = -mk i := rfl


@[simp]
theorem val_neg {x} : ∀ y : HomogeneousLocalization 𝒜 x, (-y).val = -y.val :=
                           /-
                             ι : Type u_1
                             R : Type u_2
                             A : Type u_3
                             inst✝² : CommRing R
                             inst✝¹ : CommRing A
                             inst✝ : Algebra R A
                             𝒜 : ι → Submodule R A
                             x : Submonoid A
                             y : HomogeneousLocalization.NumDenSameDeg 𝒜 x
                             ⊢ Eq (Neg.neg (Quotient.mk'' y)).val (Neg.neg (HomogeneousLocalization.val (Qu …
                           -/
  Quotient.ind' fun y ↦ by rw [← mk_neg, val_mk, val_mk, Localization.neg_mk]; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


instance hasPow : Pow (HomogeneousLocalization 𝒜 x) ℕ where
  pow z n :=
    (Quotient.map' (· ^ n) fun c1 c2 (h : Localization.mk _ _ = Localization.mk _ _) => by
          /-
            ι : Type u_1
            R : Type u_2
            A : Type u_3
            inst✝⁵ : CommRing R
            inst✝⁴ : CommRing A
            inst✝³ : Algebra R A
            𝒜 : ι → Submodule R A
            x : Submonoid A
            inst✝² : AddCommMonoid ι
            inst✝¹ : DecidableEq ι
            inst✝ : GradedAlgebra 𝒜
            z : HomogeneousLocalization 𝒜 x
            n : Nat
            c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
            h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
            ⊢ (Setoid.ker (HomogeneousLocalization.NumDenSameDeg.embedding 𝒜 x)) ((fun x_1 …
          -/
          change Localization.mk _ _ = Localization.mk _ _
          /-
            ι : Type u_1
            R : Type u_2
            A : Type u_3
            inst✝⁵ : CommRing R
            inst✝⁴ : CommRing A
            inst✝³ : Algebra R A
            𝒜 : ι → Submodule R A
            x : Submonoid A
            inst✝² : AddCommMonoid ι
            inst✝¹ : DecidableEq ι
            inst✝ : GradedAlgebra 𝒜
            z : HomogeneousLocalization 𝒜 x
            n : Nat
            c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
            h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
            ⊢ Eq (Localization.mk ↑((fun x_1 => HPow.hPow x_1 n) c1).num ⟨↑((fun x_1 => HP …
          -/
          simp only [num_pow, den_pow]
          /-
            ι : Type u_1
            R : Type u_2
            A : Type u_3
            inst✝⁵ : CommRing R
            inst✝⁴ : CommRing A
            inst✝³ : Algebra R A
            𝒜 : ι → Submodule R A
            x : Submonoid A
            inst✝² : AddCommMonoid ι
            inst✝¹ : DecidableEq ι
            inst✝ : GradedAlgebra 𝒜
            z : HomogeneousLocalization 𝒜 x
            n : Nat
            c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
            h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
            ⊢ Eq (Localization.mk (HPow.hPow (↑c1.num) n) ⟨HPow.hPow (↑c1.den) n, ⋯⟩) (Loc …
          -/
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
          convert congr_arg (fun z : at x => z ^ n) h <;> rw [Localization.mk_pow] <;> rfl :
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
        HomogeneousLocalization 𝒜 x → HomogeneousLocalization 𝒜 x)
      z


@[simp] lemma mk_pow (i : NumDenSameDeg 𝒜 x) (n : ℕ) : mk (i ^ n) = mk i ^ n := rfl


instance : Add (HomogeneousLocalization 𝒜 x) where
  add :=
    Quotient.map₂ (· + ·)
      fun c1 c2 (h : Localization.mk _ _ = Localization.mk _ _) c3 c4
        (h' : Localization.mk _ _ = Localization.mk _ _) => by
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        𝒜 : ι → Submodule R A
        x : Submonoid A
        inst✝² : AddCommMonoid ι
        inst✝¹ : DecidableEq ι
        inst✝ : GradedAlgebra 𝒜
        c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
        c3 c4 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h' : Eq (Localization.mk ↑c3.num ⟨↑c3.den, ⋯⟩) (Localization.mk ↑c4.num ⟨↑c4.d …
        ⊢ HasEquiv.Equiv ((fun x1 x2 => HAdd.hAdd x1 x2) c1 c3) ((fun x1 x2 => HAdd.hA …
      -/
      change Localization.mk _ _ = Localization.mk _ _
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        𝒜 : ι → Submodule R A
        x : Submonoid A
        inst✝² : AddCommMonoid ι
        inst✝¹ : DecidableEq ι
        inst✝ : GradedAlgebra 𝒜
        c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
        c3 c4 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h' : Eq (Localization.mk ↑c3.num ⟨↑c3.den, ⋯⟩) (Localization.mk ↑c4.num ⟨↑c4.d …
        ⊢ Eq (Localization.mk ↑((fun x1 x2 => HAdd.hAdd x1 x2) c1 c3).num ⟨↑((fun x1 x …
      -/
      simp only [num_add, den_add, ← Localization.add_mk]
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        𝒜 : ι → Submodule R A
        x : Submonoid A
        inst✝² : AddCommMonoid ι
        inst✝¹ : DecidableEq ι
        inst✝ : GradedAlgebra 𝒜
        c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
        c3 c4 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h' : Eq (Localization.mk ↑c3.num ⟨↑c3.den, ⋯⟩) (Localization.mk ↑c4.num ⟨↑c4.d …
        ⊢ Eq (Localization.mk (HAdd.hAdd (HMul.hMul ↑c1.den ↑c3.num) (HMul.hMul ↑c3.de …
      -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
      convert congr_arg₂ (· + ·) h h' <;> rw [Localization.add_mk] <;> rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp] lemma mk_add (i j : NumDenSameDeg 𝒜 x) : mk (i + j) = mk i + mk j := rfl


instance : Sub (HomogeneousLocalization 𝒜 x) where sub z1 z2 := z1 + -z2


instance : Mul (HomogeneousLocalization 𝒜 x) where
  mul :=
    Quotient.map₂ (· * ·)
      fun c1 c2 (h : Localization.mk _ _ = Localization.mk _ _) c3 c4
        (h' : Localization.mk _ _ = Localization.mk _ _) => by
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        𝒜 : ι → Submodule R A
        x : Submonoid A
        inst✝² : AddCommMonoid ι
        inst✝¹ : DecidableEq ι
        inst✝ : GradedAlgebra 𝒜
        c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
        c3 c4 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h' : Eq (Localization.mk ↑c3.num ⟨↑c3.den, ⋯⟩) (Localization.mk ↑c4.num ⟨↑c4.d …
        ⊢ HasEquiv.Equiv ((fun x1 x2 => HMul.hMul x1 x2) c1 c3) ((fun x1 x2 => HMul.hM …
      -/
      change Localization.mk _ _ = Localization.mk _ _
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        𝒜 : ι → Submodule R A
        x : Submonoid A
        inst✝² : AddCommMonoid ι
        inst✝¹ : DecidableEq ι
        inst✝ : GradedAlgebra 𝒜
        c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
        c3 c4 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h' : Eq (Localization.mk ↑c3.num ⟨↑c3.den, ⋯⟩) (Localization.mk ↑c4.num ⟨↑c4.d …
        ⊢ Eq (Localization.mk ↑((fun x1 x2 => HMul.hMul x1 x2) c1 c3).num ⟨↑((fun x1 x …
      -/
      simp only [num_mul, den_mul]
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        𝒜 : ι → Submodule R A
        x : Submonoid A
        inst✝² : AddCommMonoid ι
        inst✝¹ : DecidableEq ι
        inst✝ : GradedAlgebra 𝒜
        c1 c2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h : Eq (Localization.mk ↑c1.num ⟨↑c1.den, ⋯⟩) (Localization.mk ↑c2.num ⟨↑c2.de …
        c3 c4 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
        h' : Eq (Localization.mk ↑c3.num ⟨↑c3.den, ⋯⟩) (Localization.mk ↑c4.num ⟨↑c4.d …
        ⊢ Eq (Localization.mk (HMul.hMul ↑c1.num ↑c3.num) ⟨HMul.hMul ↑c1.den ↑c3.den,  …
      -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
      convert congr_arg₂ (· * ·) h h' <;> rw [Localization.mk_mul] <;> rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp] lemma mk_mul (i j : NumDenSameDeg 𝒜 x) : mk (i * j) = mk i * mk j := rfl


instance : One (HomogeneousLocalization 𝒜 x) where one := Quotient.mk'' 1


@[simp] lemma mk_one : mk (1 : NumDenSameDeg 𝒜 x) = 1 := rfl


instance : Zero (HomogeneousLocalization 𝒜 x) where zero := Quotient.mk'' 0


@[simp] lemma mk_zero : mk (0 : NumDenSameDeg 𝒜 x) = 0 := rfl


theorem zero_eq : (0 : HomogeneousLocalization 𝒜 x) = Quotient.mk'' 0 :=
  rfl


theorem one_eq : (1 : HomogeneousLocalization 𝒜 x) = Quotient.mk'' 1 :=
  rfl


@[simp]
theorem val_zero : (0 : HomogeneousLocalization 𝒜 x).val = 0 :=
  Localization.mk_zero _


@[simp]
theorem val_one : (1 : HomogeneousLocalization 𝒜 x).val = 1 :=
  Localization.mk_one


@[simp]
theorem val_add : ∀ y1 y2 : HomogeneousLocalization 𝒜 x, (y1 + y2).val = y1.val + y2.val :=
                                /-
                                  ι : Type u_1
                                  R : Type u_2
                                  A : Type u_3
                                  inst✝⁵ : CommRing R
                                  inst✝⁴ : CommRing A
                                  inst✝³ : Algebra R A
                                  𝒜 : ι → Submodule R A
                                  x : Submonoid A
                                  inst✝² : AddCommMonoid ι
                                  inst✝¹ : DecidableEq ι
                                  inst✝ : GradedAlgebra 𝒜
                                  y1 y2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
                                  ⊢ Eq (HAdd.hAdd (Quotient.mk'' y1) (Quotient.mk'' y2)).val (HAdd.hAdd (Homogen …
                                -/
  Quotient.ind₂' fun y1 y2 ↦ by rw [← mk_add, val_mk, val_mk, val_mk, Localization.add_mk]; rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[simp]
theorem val_mul : ∀ y1 y2 : HomogeneousLocalization 𝒜 x, (y1 * y2).val = y1.val * y2.val :=
                                /-
                                  ι : Type u_1
                                  R : Type u_2
                                  A : Type u_3
                                  inst✝⁵ : CommRing R
                                  inst✝⁴ : CommRing A
                                  inst✝³ : Algebra R A
                                  𝒜 : ι → Submodule R A
                                  x : Submonoid A
                                  inst✝² : AddCommMonoid ι
                                  inst✝¹ : DecidableEq ι
                                  inst✝ : GradedAlgebra 𝒜
                                  y1 y2 : HomogeneousLocalization.NumDenSameDeg 𝒜 x
                                  ⊢ Eq (HMul.hMul (Quotient.mk'' y1) (Quotient.mk'' y2)).val (HMul.hMul (Homogen …
                                -/
  Quotient.ind₂' fun y1 y2 ↦ by rw [← mk_mul, val_mk, val_mk, val_mk, Localization.mk_mul]; rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[simp]
theorem val_sub (y1 y2 : HomogeneousLocalization 𝒜 x) : (y1 - y2).val = y1.val - y2.val := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    y1 y2 : HomogeneousLocalization 𝒜 x
    ⊢ Eq (HSub.hSub y1 y2).val (HSub.hSub y1.val y2.val)
  -/
  rw [sub_eq_add_neg, ← val_neg, ← val_add]; rfl
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem val_pow : ∀ (y : HomogeneousLocalization 𝒜 x) (n : ℕ), (y ^ n).val = y.val ^ n :=
                             /-
                               ι : Type u_1
                               R : Type u_2
                               A : Type u_3
                               inst✝⁵ : CommRing R
                               inst✝⁴ : CommRing A
                               inst✝³ : Algebra R A
                               𝒜 : ι → Submodule R A
                               x : Submonoid A
                               inst✝² : AddCommMonoid ι
                               inst✝¹ : DecidableEq ι
                               inst✝ : GradedAlgebra 𝒜
                               y : HomogeneousLocalization.NumDenSameDeg 𝒜 x
                               n : Nat
                               ⊢ Eq (HPow.hPow (Quotient.mk'' y) n).val (HPow.hPow (HomogeneousLocalization.v …
                             -/
  Quotient.ind' fun y n ↦ by rw [← mk_pow, val_mk, val_mk, Localization.mk_pow]; rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


instance : NatCast (HomogeneousLocalization 𝒜 x) :=
  ⟨Nat.unaryCast⟩


instance : IntCast (HomogeneousLocalization 𝒜 x) :=
  ⟨Int.castDef⟩


@[simp]
theorem val_natCast (n : ℕ) : (n : HomogeneousLocalization 𝒜 x).val = n :=
                                    /-
                                      ι : Type u_1
                                      R : Type u_2
                                      A : Type u_3
                                      inst✝⁵ : CommRing R
                                      inst✝⁴ : CommRing A
                                      inst✝³ : Algebra R A
                                      𝒜 : ι → Submodule R A
                                      x : Submonoid A
                                      inst✝² : AddCommMonoid ι
                                      inst✝¹ : DecidableEq ι
                                      inst✝ : GradedAlgebra 𝒜
                                      n : Nat
                                      ⊢ Eq n.unaryCast.val ↑n
                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  show val (Nat.unaryCast n) = _ by induction n <;> simp [Nat.unaryCast, *]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem val_intCast (n : ℤ) : (n : HomogeneousLocalization 𝒜 x).val = n :=
                                  /-
                                    ι : Type u_1
                                    R : Type u_2
                                    A : Type u_3
                                    inst✝⁵ : CommRing R
                                    inst✝⁴ : CommRing A
                                    inst✝³ : Algebra R A
                                    𝒜 : ι → Submodule R A
                                    x : Submonoid A
                                    inst✝² : AddCommMonoid ι
                                    inst✝¹ : DecidableEq ι
                                    inst✝ : GradedAlgebra 𝒜
                                    n : Int
                                    ⊢ Eq n.castDef.val ↑n
                                  -/
                                              /-
                                                🎉 no goals
                                              -/
  show val (Int.castDef n) = _ by cases n <;> simp [Int.castDef, *]
                                              /-
                                                🎉 no goals
                                              -/


instance homogeneousLocalizationCommRing : CommRing (HomogeneousLocalization 𝒜 x) :=
  (HomogeneousLocalization.val_injective x).commRing _ val_zero val_one val_add val_mul val_neg
    val_sub (val_nsmul x · ·) (val_zsmul x · ·) val_pow val_natCast val_intCast


instance homogeneousLocalizationAlgebra :
    Algebra (HomogeneousLocalization 𝒜 x) (Localization x) where
  smul p q := p.val * q
  toFun := val
  map_one' := val_one
  map_mul' := val_mul
  map_zero' := val_zero
  map_add' := val_add
  commutes' _ _ := mul_comm _ _
  smul_def' _ _ := rfl


@[simp] lemma algebraMap_apply (y) :
    algebraMap (HomogeneousLocalization 𝒜 x) (Localization x) y = y.val := rfl


lemma mk_eq_zero_of_num (f : NumDenSameDeg 𝒜 x) (h : f.num = 0) : mk f = 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    f : HomogeneousLocalization.NumDenSameDeg 𝒜 x
    h : Eq f.num 0
    ⊢ Eq (HomogeneousLocalization.mk f) 0
  -/
  apply val_injective
  /-
    case a
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    f : HomogeneousLocalization.NumDenSameDeg 𝒜 x
    h : Eq f.num 0
    ⊢ Eq (HomogeneousLocalization.mk f).val (HomogeneousLocalization.val 0)
  -/
  simp only [val_mk, val_zero, h, ZeroMemClass.coe_zero, Localization.mk_zero]
  /-
    🎉 no goals
  -/


lemma mk_eq_zero_of_den (f : NumDenSameDeg 𝒜 x) (h : f.den = 0) : mk f = 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    f : HomogeneousLocalization.NumDenSameDeg 𝒜 x
    h : Eq f.den 0
    ⊢ Eq (HomogeneousLocalization.mk f) 0
  -/
  have := subsingleton 𝒜 (h ▸ f.den_mem)
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    f : HomogeneousLocalization.NumDenSameDeg 𝒜 x
    h : Eq f.den 0
    this : Subsingleton (HomogeneousLocalization 𝒜 x)
    ⊢ Eq (HomogeneousLocalization.mk f) 0
  -/
  exact Subsingleton.elim _ _
  /-
    🎉 no goals
  -/


variable (𝒜 x) in
/-- The map from `𝒜 0` to the degree `0` part of `𝒜ₓ` sending `f ↦ f/1`. -/
def fromZeroRingHom : 𝒜 0 →+* HomogeneousLocalization 𝒜 x where
  toFun f := .mk ⟨0, f, 1, one_mem _⟩
  map_one' := rfl
                     /-
                       ι : Type u_1
                       R : Type u_2
                       A : Type u_3
                       inst✝⁵ : CommRing R
                       inst✝⁴ : CommRing A
                       inst✝³ : Algebra R A
                       𝒜 : ι → Submodule R A
                       x : Submonoid A
                       inst✝² : AddCommMonoid ι
                       inst✝¹ : DecidableEq ι
                       inst✝ : GradedAlgebra 𝒜
                       f g : Subtype fun x => Membership.mem (𝒜 0) x
                       ⊢ Eq ({ toFun := fun f => HomogeneousLocalization.mk { deg := 0, num := f, den …
                     -/
  map_mul' f g := by ext; simp [Localization.mk_mul]
                          /-
                            🎉 no goals
                          -/
  map_zero' := rfl
                     /-
                       ι : Type u_1
                       R : Type u_2
                       A : Type u_3
                       inst✝⁵ : CommRing R
                       inst✝⁴ : CommRing A
                       inst✝³ : Algebra R A
                       𝒜 : ι → Submodule R A
                       x : Submonoid A
                       inst✝² : AddCommMonoid ι
                       inst✝¹ : DecidableEq ι
                       inst✝ : GradedAlgebra 𝒜
                       f g : Subtype fun x => Membership.mem (𝒜 0) x
                       ⊢ Eq ((↑{ toFun := fun f => HomogeneousLocalization.mk { deg := 0, num := f, d …
                     -/
  map_add' f g := by ext; simp [Localization.add_mk, add_comm f.1 g.1]
                          /-
                            🎉 no goals
                          -/


instance : Algebra (𝒜 0) (HomogeneousLocalization 𝒜 x) :=
  (fromZeroRingHom 𝒜 x).toAlgebra


lemma algebraMap_eq : algebraMap (𝒜 0) (HomogeneousLocalization 𝒜 x) = fromZeroRingHom 𝒜 x := rfl


/-- Numerator of an element in `HomogeneousLocalization x`. -/
def num (f : HomogeneousLocalization 𝒜 x) : A :=
  (Quotient.out f).num


/-- Denominator of an element in `HomogeneousLocalization x`. -/
def den (f : HomogeneousLocalization 𝒜 x) : A :=
  (Quotient.out f).den


/-- For an element in `HomogeneousLocalization x`, degree is the natural number `i` such that
  `𝒜 i` contains both numerator and denominator. -/
def deg (f : HomogeneousLocalization 𝒜 x) : ι :=
  (Quotient.out f).deg


theorem den_mem (f : HomogeneousLocalization 𝒜 x) : f.den ∈ x :=
  (Quotient.out f).den_mem


theorem num_mem_deg (f : HomogeneousLocalization 𝒜 x) : f.num ∈ 𝒜 f.deg :=
  (Quotient.out f).num.2


theorem den_mem_deg (f : HomogeneousLocalization 𝒜 x) : f.den ∈ 𝒜 f.deg :=
  (Quotient.out f).den.2


theorem eq_num_div_den (f : HomogeneousLocalization 𝒜 x) :
    f.val = Localization.mk f.num ⟨f.den, f.den_mem⟩ :=
  congr_arg HomogeneousLocalization.val (Quotient.out_eq' f).symm


theorem den_smul_val (f : HomogeneousLocalization 𝒜 x) :
    f.den • f.val = algebraMap _ _ f.num := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    f : HomogeneousLocalization 𝒜 x
    ⊢ Eq (HSMul.hSMul f.den f.val) ((algebraMap A (Localization x)) f.num)
  -/
  rw [eq_num_div_den, Localization.mk_eq_mk', IsLocalization.smul_mk']
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    𝒜 : ι → Submodule R A
    x : Submonoid A
    f : HomogeneousLocalization 𝒜 x
    ⊢ Eq (IsLocalization.mk' (Localization x) (HMul.hMul f.den f.num) ⟨f.den, ⋯⟩)  …
  -/
  exact IsLocalization.mk'_mul_cancel_left _ ⟨_, _⟩
  /-
    🎉 no goals
  -/


theorem ext_iff_val (f g : HomogeneousLocalization 𝒜 x) : f = g ↔ f.val = g.val :=
  ⟨congr_arg val, fun e ↦ val_injective x e⟩


/-- Localizing a ring homogeneously at a prime ideal. -/
abbrev AtPrime :=
  HomogeneousLocalization 𝒜 𝔭.primeCompl


theorem isUnit_iff_isUnit_val (f : HomogeneousLocalization.AtPrime 𝒜 𝔭) :
    IsUnit f.val ↔ IsUnit f := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝³ : AddCommMonoid ι
    inst✝² : DecidableEq ι
    inst✝¹ : GradedAlgebra 𝒜
    𝔭 : Ideal A
    inst✝ : 𝔭.IsPrime
    f : HomogeneousLocalization.AtPrime 𝒜 𝔭
    ⊢ Iff (IsUnit (HomogeneousLocalization.val f)) (IsUnit f)
  -/
  refine ⟨fun h1 ↦ ?_, IsUnit.map (algebraMap _ _)⟩
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝³ : AddCommMonoid ι
    inst✝² : DecidableEq ι
    inst✝¹ : GradedAlgebra 𝒜
    𝔭 : Ideal A
    inst✝ : 𝔭.IsPrime
    f : HomogeneousLocalization.AtPrime 𝒜 𝔭
    h1 : IsUnit (HomogeneousLocalization.val f)
    ⊢ IsUnit f
  -/
  rcases h1 with ⟨⟨a, b, eq0, eq1⟩, rfl : a = f.val⟩
  /-
    case intro.mk
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝³ : AddCommMonoid ι
    inst✝² : DecidableEq ι
    inst✝¹ : GradedAlgebra 𝒜
    𝔭 : Ideal A
    inst✝ : 𝔭.IsPrime
    f : HomogeneousLocalization.AtPrime 𝒜 𝔭
    b : Localization 𝔭.primeCompl
    eq0 : Eq (HMul.hMul (HomogeneousLocalization.val f) b) 1
    eq1 : Eq (HMul.hMul b (HomogeneousLocalization.val f)) 1
    ⊢ IsUnit f
  -/
  obtain ⟨f, rfl⟩ := mk_surjective f
  /-
    case intro.mk.intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝³ : AddCommMonoid ι
    inst✝² : DecidableEq ι
    inst✝¹ : GradedAlgebra 𝒜
    𝔭 : Ideal A
    inst✝ : 𝔭.IsPrime
    b : Localization 𝔭.primeCompl
    f : HomogeneousLocalization.NumDenSameDeg 𝒜 𝔭.primeCompl
    eq0 : Eq (HMul.hMul (HomogeneousLocalization.mk f).val b) 1
    eq1 : Eq (HMul.hMul b (HomogeneousLocalization.mk f).val) 1
    ⊢ IsUnit (HomogeneousLocalization.mk f)
  -/
  obtain ⟨b, s, rfl⟩ := IsLocalization.mk'_surjective 𝔭.primeCompl b
  rw [val_mk, Localization.mk_eq_mk', ← IsLocalization.mk'_mul, IsLocalization.mk'_eq_iff_eq_mul,
    one_mul, IsLocalization.eq_iff_exists (M := 𝔭.primeCompl)] at eq0
  /-
    case intro.mk.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝³ : AddCommMonoid ι
    inst✝² : DecidableEq ι
    inst✝¹ : GradedAlgebra 𝒜
    𝔭 : Ideal A
    inst✝ : 𝔭.IsPrime
    f : HomogeneousLocalization.NumDenSameDeg 𝒜 𝔭.primeCompl
    b : A
    s : Subtype fun x => Membership.mem 𝔭.primeCompl x
    eq0 : Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul (↑f.num) b)) (HMul.hMul ↑c …
    eq1 : Eq (HMul.hMul (IsLocalization.mk' (Localization 𝔭.primeCompl) b s) (Homo …
    ⊢ IsUnit (HomogeneousLocalization.mk f)
  -/
  obtain ⟨c, hc : _ = c.1 * (f.den.1 * s.1)⟩ := eq0
  have : f.num.1 ∉ 𝔭 := by
    exact fun h ↦ mul_mem c.2 (mul_mem f.den_mem s.2)
      (hc ▸ Ideal.mul_mem_left _ c.1 (Ideal.mul_mem_right b _ h))
  /-
    case intro.mk.intro.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝³ : AddCommMonoid ι
    inst✝² : DecidableEq ι
    inst✝¹ : GradedAlgebra 𝒜
    𝔭 : Ideal A
    inst✝ : 𝔭.IsPrime
    f : HomogeneousLocalization.NumDenSameDeg 𝒜 𝔭.primeCompl
    b : A
    s : Subtype fun x => Membership.mem 𝔭.primeCompl x
    eq1 : Eq (HMul.hMul (IsLocalization.mk' (Localization 𝔭.primeCompl) b s) (Homo …
    c : Subtype fun x => Membership.mem 𝔭.primeCompl x
    hc : Eq (HMul.hMul (↑c) (HMul.hMul (↑f.num) b)) (HMul.hMul (↑c) (HMul.hMul ↑f. …
    this : Not (Membership.mem 𝔭 ↑f.num)
    ⊢ IsUnit (HomogeneousLocalization.mk f)
  -/
  refine isUnit_of_mul_eq_one _ (Quotient.mk'' ⟨f.1, f.3, f.2, this⟩) ?_
  /-
    case intro.mk.intro.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝³ : AddCommMonoid ι
    inst✝² : DecidableEq ι
    inst✝¹ : GradedAlgebra 𝒜
    𝔭 : Ideal A
    inst✝ : 𝔭.IsPrime
    f : HomogeneousLocalization.NumDenSameDeg 𝒜 𝔭.primeCompl
    b : A
    s : Subtype fun x => Membership.mem 𝔭.primeCompl x
    eq1 : Eq (HMul.hMul (IsLocalization.mk' (Localization 𝔭.primeCompl) b s) (Homo …
    c : Subtype fun x => Membership.mem 𝔭.primeCompl x
    hc : Eq (HMul.hMul (↑c) (HMul.hMul (↑f.num) b)) (HMul.hMul (↑c) (HMul.hMul ↑f. …
    this : Not (Membership.mem 𝔭 ↑f.num)
    ⊢ Eq (HMul.hMul (HomogeneousLocalization.mk f) (Quotient.mk'' { deg := f.deg,  …
  -/
  rw [← mk_mul, ext_iff_val, val_mk]
  /-
    case intro.mk.intro.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝³ : AddCommMonoid ι
    inst✝² : DecidableEq ι
    inst✝¹ : GradedAlgebra 𝒜
    𝔭 : Ideal A
    inst✝ : 𝔭.IsPrime
    f : HomogeneousLocalization.NumDenSameDeg 𝒜 𝔭.primeCompl
    b : A
    s : Subtype fun x => Membership.mem 𝔭.primeCompl x
    eq1 : Eq (HMul.hMul (IsLocalization.mk' (Localization 𝔭.primeCompl) b s) (Homo …
    c : Subtype fun x => Membership.mem 𝔭.primeCompl x
    hc : Eq (HMul.hMul (↑c) (HMul.hMul (↑f.num) b)) (HMul.hMul (↑c) (HMul.hMul ↑f. …
    this : Not (Membership.mem 𝔭 ↑f.num)
    ⊢ Eq (Localization.mk ↑(HMul.hMul f { deg := f.deg, num := f.den, den := f.num …
  -/
  simp [mul_comm f.den.1, Localization.mk_eq_monoidOf_mk']
  /-
    🎉 no goals
  -/


instance : Nontrivial (HomogeneousLocalization.AtPrime 𝒜 𝔭) :=
                      /-
                        ι : Type u_1
                        R : Type u_2
                        A : Type u_3
                        inst✝⁶ : CommRing R
                        inst✝⁵ : CommRing A
                        inst✝⁴ : Algebra R A
                        𝒜 : ι → Submodule R A
                        x : Submonoid A
                        inst✝³ : AddCommMonoid ι
                        inst✝² : DecidableEq ι
                        inst✝¹ : GradedAlgebra 𝒜
                        𝔭 : Ideal A
                        inst✝ : 𝔭.IsPrime
                        r : Eq 0 1
                        ⊢ False
                      -/
  ⟨⟨0, 1, fun r => by simp [ext_iff_val, val_zero, val_one, zero_ne_one] at r⟩⟩
                      /-
                        🎉 no goals
                      -/


instance isLocalRing : IsLocalRing (HomogeneousLocalization.AtPrime 𝒜 𝔭) :=
  IsLocalRing.of_isUnit_or_isUnit_one_sub_self fun a => by
    simpa only [← isUnit_iff_isUnit_val, val_sub, val_one]
      using IsLocalRing.isUnit_or_isUnit_one_sub_self _


/-- Localizing away from powers of `f` homogeneously. -/
abbrev Away :=
  HomogeneousLocalization 𝒜 (Submonoid.powers f)


theorem Away.eventually_smul_mem {m} (hf : f ∈ 𝒜 m) (z : Away 𝒜 f) :
    ∀ᶠ n in Filter.atTop, f ^ n • z.val ∈ algebraMap _ _ '' (𝒜 (n • m) : Set A) := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    f : A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    m : ι
    hf : Membership.mem (𝒜 m) f
    z : HomogeneousLocalization.Away 𝒜 f
    ⊢ Filter.Eventually (fun n => Membership.mem (Set.image ⇑(algebraMap A (Locali …
  -/
  obtain ⟨k, hk : f ^ k = _⟩ := z.den_mem
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    f : A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    m : ι
    hf : Membership.mem (𝒜 m) f
    z : HomogeneousLocalization.Away 𝒜 f
    k : Nat
    hk : Eq (HPow.hPow f k) (HomogeneousLocalization.den z)
    ⊢ Filter.Eventually (fun n => Membership.mem (Set.image ⇑(algebraMap A (Locali …
  -/
  apply Filter.mem_of_superset (Filter.Ici_mem_atTop k)
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    f : A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    m : ι
    hf : Membership.mem (𝒜 m) f
    z : HomogeneousLocalization.Away 𝒜 f
    k : Nat
    hk : Eq (HPow.hPow f k) (HomogeneousLocalization.den z)
    ⊢ HasSubset.Subset (Set.Ici k) (setOf fun x => (fun n => Membership.mem (Set.i …
  -/
  rintro k' (hk' : k ≤ k')
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    f : A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    m : ι
    hf : Membership.mem (𝒜 m) f
    z : HomogeneousLocalization.Away 𝒜 f
    k : Nat
    hk : Eq (HPow.hPow f k) (HomogeneousLocalization.den z)
    k' : Nat
    hk' : LE.le k k'
    ⊢ Membership.mem (setOf fun x => (fun n => Membership.mem (Set.image ⇑(algebra …
  -/
  simp only [Set.mem_image, SetLike.mem_coe, Set.mem_setOf_eq]
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    f : A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    m : ι
    hf : Membership.mem (𝒜 m) f
    z : HomogeneousLocalization.Away 𝒜 f
    k : Nat
    hk : Eq (HPow.hPow f k) (HomogeneousLocalization.den z)
    k' : Nat
    hk' : LE.le k k'
    ⊢ Exists fun x => And (Membership.mem (𝒜 (HSMul.hSMul k' m)) x) (Eq ((algebraM …
  -/
  by_cases hfk : f ^ k = 0
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      𝒜 : ι → Submodule R A
      f : A
      inst✝² : AddCommMonoid ι
      inst✝¹ : DecidableEq ι
      inst✝ : GradedAlgebra 𝒜
      m : ι
      hf : Membership.mem (𝒜 m) f
      z : HomogeneousLocalization.Away 𝒜 f
      k : Nat
      hk : Eq (HPow.hPow f k) (HomogeneousLocalization.den z)
      k' : Nat
      hk' : LE.le k k'
      hfk : Eq (HPow.hPow f k) 0
      ⊢ Exists fun x => And (Membership.mem (𝒜 (HSMul.hSMul k' m)) x) (Eq ((algebraM …
    -/
  · refine ⟨0, zero_mem _, ?_⟩
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      𝒜 : ι → Submodule R A
      f : A
      inst✝² : AddCommMonoid ι
      inst✝¹ : DecidableEq ι
      inst✝ : GradedAlgebra 𝒜
      m : ι
      hf : Membership.mem (𝒜 m) f
      z : HomogeneousLocalization.Away 𝒜 f
      k : Nat
      hk : Eq (HPow.hPow f k) (HomogeneousLocalization.den z)
      k' : Nat
      hk' : LE.le k k'
      hfk : Eq (HPow.hPow f k) 0
      ⊢ Eq ((algebraMap A (Localization (Submonoid.powers f))) 0) (HSMul.hSMul (HPow …
    -/
    rw [← tsub_add_cancel_of_le hk', map_zero, pow_add, hfk, mul_zero, zero_smul]
    /-
      🎉 no goals
    -/
  rw [← tsub_add_cancel_of_le hk', pow_add, mul_smul, hk, den_smul_val,
    Algebra.smul_def, ← _root_.map_mul]
  rw [← smul_eq_mul, add_smul,
    DirectSum.degree_eq_of_mem_mem 𝒜 (SetLike.pow_mem_graded _ hf) (hk.symm ▸ z.den_mem_deg) hfk]
  /-
    case neg
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    f : A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    m : ι
    hf : Membership.mem (𝒜 m) f
    z : HomogeneousLocalization.Away 𝒜 f
    k : Nat
    hk : Eq (HPow.hPow f k) (HomogeneousLocalization.den z)
    k' : Nat
    hk' : LE.le k k'
    hfk : Not (Eq (HPow.hPow f k) 0)
    ⊢ Exists fun x => And (Membership.mem (𝒜 (HAdd.hAdd (HSMul.hSMul (HSub.hSub k' …
  -/
  exact ⟨_, SetLike.mul_mem_graded (SetLike.pow_mem_graded _ hf) z.num_mem_deg, rfl⟩
  /-
    🎉 no goals
  -/


/--
Let `A, B` be two graded algebras with the same indexing set and `g : A → B` be a graded algebra
homomorphism (i.e. `g(Aₘ) ⊆ Bₘ`). Let `P ≤ A` be a submonoid and `Q ≤ B` be a submonoid such that
`P ≤ g⁻¹ Q`, then `g` induce a map from the homogeneous localizations `A⁰_P` to the homogeneous
localizations `B⁰_Q`.
-/
def map (g : A →+* B)
    (comap_le : P ≤ Q.comap g) (hg : ∀ i, ∀ a ∈ 𝒜 i, g a ∈ ℬ i) :
    HomogeneousLocalization 𝒜 P →+* HomogeneousLocalization ℬ Q where
  toFun := Quotient.map'
    (fun x ↦ ⟨x.1, ⟨_, hg _ _ x.2.2⟩, ⟨_, hg _ _ x.3.2⟩, comap_le x.4⟩)
    fun x y (e : x.embedding = y.embedding) ↦ by
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing A
        inst✝⁶ : Algebra R A
        𝒜 : ι → Submodule R A
        x✝ : Submonoid A
        inst✝⁵ : AddCommMonoid ι
        inst✝⁴ : DecidableEq ι
        inst✝³ : GradedAlgebra 𝒜
        B : Type u_4
        inst✝² : CommRing B
        inst✝¹ : Algebra R B
        ℬ : ι → Submodule R B
        inst✝ : GradedAlgebra ℬ
        P : Submonoid A
        Q : Submonoid B
        g : RingHom A B
        comap_le : LE.le P (Submonoid.comap g Q)
        hg : ∀ (i : ι) (a : A), Membership.mem (𝒜 i) a → Membership.mem (ℬ i) (g a)
        x y : HomogeneousLocalization.NumDenSameDeg 𝒜 P
        e : Eq (HomogeneousLocalization.NumDenSameDeg.embedding 𝒜 P x) (HomogeneousLoc …
        ⊢ (Setoid.ker (HomogeneousLocalization.NumDenSameDeg.embedding ℬ Q)) ((fun x = …
      -/
      apply_fun IsLocalization.map (Localization Q) g comap_le at e
      simp_rw [HomogeneousLocalization.NumDenSameDeg.embedding, Localization.mk_eq_mk',
        IsLocalization.map_mk', ← Localization.mk_eq_mk'] at e
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing A
        inst✝⁶ : Algebra R A
        𝒜 : ι → Submodule R A
        x✝ : Submonoid A
        inst✝⁵ : AddCommMonoid ι
        inst✝⁴ : DecidableEq ι
        inst✝³ : GradedAlgebra 𝒜
        B : Type u_4
        inst✝² : CommRing B
        inst✝¹ : Algebra R B
        ℬ : ι → Submodule R B
        inst✝ : GradedAlgebra ℬ
        P : Submonoid A
        Q : Submonoid B
        g : RingHom A B
        comap_le : LE.le P (Submonoid.comap g Q)
        hg : ∀ (i : ι) (a : A), Membership.mem (𝒜 i) a → Membership.mem (ℬ i) (g a)
        x y : HomogeneousLocalization.NumDenSameDeg 𝒜 P
        e : Eq (Localization.mk (g ↑x.num) ⟨g ↑x.den, ⋯⟩) (Localization.mk (g ↑y.num)  …
        ⊢ (Setoid.ker (HomogeneousLocalization.NumDenSameDeg.embedding ℬ Q)) ((fun x = …
      -/
      exact e
      /-
        🎉 no goals
      -/
  map_add' := Quotient.ind₂' fun x y ↦ by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing A
      inst✝⁶ : Algebra R A
      𝒜 : ι → Submodule R A
      x✝ : Submonoid A
      inst✝⁵ : AddCommMonoid ι
      inst✝⁴ : DecidableEq ι
      inst✝³ : GradedAlgebra 𝒜
      B : Type u_4
      inst✝² : CommRing B
      inst✝¹ : Algebra R B
      ℬ : ι → Submodule R B
      inst✝ : GradedAlgebra ℬ
      P : Submonoid A
      Q : Submonoid B
      g : RingHom A B
      comap_le : LE.le P (Submonoid.comap g Q)
      hg : ∀ (i : ι) (a : A), Membership.mem (𝒜 i) a → Membership.mem (ℬ i) (g a)
      x y : HomogeneousLocalization.NumDenSameDeg 𝒜 P
      ⊢ Eq ((↑{ toFun := Quotient.map' (fun x => { deg := x.deg, num := ⟨g ↑x.num, ⋯ …
    -/
    simp only [← mk_add, Quotient.map'_mk'', num_add, map_add, map_mul, den_add]; rfl
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing A
      inst✝⁶ : Algebra R A
      𝒜 : ι → Submodule R A
      x✝ : Submonoid A
      inst✝⁵ : AddCommMonoid ι
      inst✝⁴ : DecidableEq ι
      inst✝³ : GradedAlgebra 𝒜
      B : Type u_4
      inst✝² : CommRing B
      inst✝¹ : Algebra R B
      ℬ : ι → Submodule R B
      inst✝ : GradedAlgebra ℬ
      P : Submonoid A
      Q : Submonoid B
      g : RingHom A B
      comap_le : LE.le P (Submonoid.comap g Q)
      hg : ∀ (i : ι) (a : A), Membership.mem (𝒜 i) a → Membership.mem (ℬ i) (g a)
      x y : HomogeneousLocalization.NumDenSameDeg 𝒜 P
      ⊢ Eq ({ toFun := Quotient.map' (fun x => { deg := x.deg, num := ⟨g ↑x.num, ⋯⟩, …
    -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  map_mul' := Quotient.ind₂' fun x y ↦ by
                                                                 /-
                                                                   ι : Type u_1
                                                                   R : Type u_2
                                                                   A : Type u_3
                                                                   inst✝⁸ : CommRing R
                                                                   inst✝⁷ : CommRing A
                                                                   inst✝⁶ : Algebra R A
                                                                   𝒜 : ι → Submodule R A
                                                                   x : Submonoid A
                                                                   inst✝⁵ : AddCommMonoid ι
                                                                   inst✝⁴ : DecidableEq ι
                                                                   inst✝³ : GradedAlgebra 𝒜
                                                                   B : Type u_4
                                                                   inst✝² : CommRing B
                                                                   inst✝¹ : Algebra R B
                                                                   ℬ : ι → Submodule R B
                                                                   inst✝ : GradedAlgebra ℬ
                                                                   P : Submonoid A
                                                                   Q : Submonoid B
                                                                   g : RingHom A B
                                                                   comap_le : LE.le P (Submonoid.comap g Q)
                                                                   hg : ∀ (i : ι) (a : A), Membership.mem (𝒜 i) a → Membership.mem (ℬ i) (g a)
                                                                   ⊢ Eq (Quotient.mk'' { deg := HomogeneousLocalization.NumDenSameDeg.deg 1, num  …
                                                                 -/
                                                                   /-
                                                                     ι : Type u_1
                                                                     R : Type u_2
                                                                     A : Type u_3
                                                                     inst✝⁸ : CommRing R
                                                                     inst✝⁷ : CommRing A
                                                                     inst✝⁶ : Algebra R A
                                                                     𝒜 : ι → Submodule R A
                                                                     x : Submonoid A
                                                                     inst✝⁵ : AddCommMonoid ι
                                                                     inst✝⁴ : DecidableEq ι
                                                                     inst✝³ : GradedAlgebra 𝒜
                                                                     B : Type u_4
                                                                     inst✝² : CommRing B
                                                                     inst✝¹ : Algebra R B
                                                                     ℬ : ι → Submodule R B
                                                                     inst✝ : GradedAlgebra ℬ
                                                                     P : Submonoid A
                                                                     Q : Submonoid B
                                                                     g : RingHom A B
                                                                     comap_le : LE.le P (Submonoid.comap g Q)
                                                                     hg : ∀ (i : ι) (a : A), Membership.mem (𝒜 i) a → Membership.mem (ℬ i) (g a)
                                                                     ⊢ Eq (Quotient.mk'' { deg := 0, num := ⟨0, ⋯⟩, den := ⟨1, ⋯⟩, den_mem := ⋯ }) 0
                                                                   -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    simp only [← mk_mul, Quotient.map'_mk'', num_mul, map_mul, den_mul]; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  map_zero' := by simp only [← mk_zero (𝒜 := 𝒜), Quotient.map'_mk'', deg_zero,
    num_zero, ZeroMemClass.coe_zero, map_zero, den_zero, map_one]; rfl
  map_one' := by simp only [← mk_one (𝒜 := 𝒜), Quotient.map'_mk'', deg_zero,
    num_one, ZeroMemClass.coe_zero, map_zero, den_one, map_one]; rfl


/--
Let `A` be a graded algebra and `P ≤ Q` be two submonoids, then the homogeneous localization of `A`
at `P` embeds into the homogeneous localization of `A` at `Q`.
-/
abbrev mapId {P Q : Submonoid A} (h : P ≤ Q) :
    HomogeneousLocalization 𝒜 P →+* HomogeneousLocalization 𝒜 Q :=
  map 𝒜 𝒜 (RingHom.id _) h (fun _ _ ↦ id)


lemma map_mk (g : A →+* B)
    (comap_le : P ≤ Q.comap g) (hg : ∀ i, ∀ a ∈ 𝒜 i, g a ∈ ℬ i) (x) :
    map 𝒜 ℬ g comap_le hg (mk x) =
    mk ⟨x.1, ⟨_, hg _ _ x.2.2⟩, ⟨_, hg _ _ x.3.2⟩, comap_le x.4⟩ :=
  rfl


/-- Given `f ∣ x`, this is the map `A_{(f)} → A_f → A_x`. We will lift this to a map
`A_{(f)} → A_{(x)}` in `awayMap`. -/
private def awayMapAux (hx : f ∣ x) : Away 𝒜 f →+* Localization.Away x :=
  (Localization.awayLift (algebraMap A _) _
    (isUnit_of_dvd_unit (map_dvd _ hx) (IsLocalization.Away.algebraMap_isUnit x))).comp
      (algebraMap (Away 𝒜 f) (Localization.Away f))


lemma awayMapAux_mk (n a i hi) :
    awayMapAux 𝒜 ⟨_, hx⟩ (mk ⟨n, a, ⟨f ^ i, hi⟩, ⟨i, rfl⟩⟩) =
      Localization.mk (a * g ^ i) ⟨x ^ i, (Submonoid.mem_powers_iff _ _).mpr ⟨i, rfl⟩⟩ := by
  have : algebraMap A (Localization.Away x) f *
    (Localization.mk g ⟨f * g, (Submonoid.mem_powers_iff _ _).mpr ⟨1, by simp [hx]⟩⟩) = 1 := by
    rw [← Algebra.smul_def, Localization.smul_mk]
    exact Localization.mk_self ⟨f*g, _⟩
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    f g x : A
    hx : Eq x (HMul.hMul f g)
    n : ι
    a : Subtype fun x => Membership.mem (𝒜 n) x
    i : Nat
    hi : Membership.mem (𝒜 n) (HPow.hPow f i)
    this : Eq (HMul.hMul ((algebraMap A (Localization.Away x)) f) (Localization.mk …
    ⊢ Eq ((HomogeneousLocalization.awayMapAux 𝒜 ⋯) (HomogeneousLocalization.mk { d …
  -/
  simp [awayMapAux]
  rw [Localization.awayLift_mk (hv := this), ← Algebra.smul_def,
    Localization.mk_pow, Localization.smul_mk]
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    f g x : A
    hx : Eq x (HMul.hMul f g)
    n : ι
    a : Subtype fun x => Membership.mem (𝒜 n) x
    i : Nat
    hi : Membership.mem (𝒜 n) (HPow.hPow f i)
    this : Eq (HMul.hMul ((algebraMap A (Localization.Away x)) f) (Localization.mk …
    ⊢ Eq (Localization.mk (HSMul.hSMul (↑a) (HPow.hPow g i)) (HPow.hPow ⟨HMul.hMul …
  -/
  subst hx
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    f g : A
    n : ι
    a : Subtype fun x => Membership.mem (𝒜 n) x
    i : Nat
    hi : Membership.mem (𝒜 n) (HPow.hPow f i)
    this : Eq (HMul.hMul ((algebraMap A (Localization.Away (HMul.hMul f g))) f) (L …
    ⊢ Eq (Localization.mk (HSMul.hSMul (↑a) (HPow.hPow g i)) (HPow.hPow ⟨HMul.hMul …
  -/
  rfl
  /-
    🎉 no goals
  -/


include hg in
lemma range_awayMapAux_subset :
    Set.range (awayMapAux 𝒜 (f := f) ⟨_, hx⟩) ⊆ Set.range (val (𝒜 := 𝒜)) := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e : ι
    f g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    ⊢ HasSubset.Subset (Set.range ⇑(HomogeneousLocalization.awayMapAux 𝒜 ⋯)) (Set. …
  -/
  rintro _ ⟨z, rfl⟩
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e : ι
    f g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    z : HomogeneousLocalization.Away 𝒜 f
    ⊢ Membership.mem (Set.range HomogeneousLocalization.val) ((HomogeneousLocaliza …
  -/
  obtain ⟨⟨n, ⟨a, ha⟩, ⟨b, hb'⟩, j, rfl : _ = b⟩, rfl⟩ := mk_surjective z
  /-
    case intro.intro.mk.mk.mk.intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e : ι
    f g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    n : ι
    a : A
    ha : Membership.mem (𝒜 n) a
    j : Nat
    hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) j)
    ⊢ Membership.mem (Set.range HomogeneousLocalization.val) ((HomogeneousLocaliza …
  -/
  use mk ⟨n+j•e,⟨a*g^j, ?_⟩ ,⟨x^j, ?_⟩, j, rfl⟩
    /-
      case h
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      𝒜 : ι → Submodule R A
      inst✝² : AddCommMonoid ι
      inst✝¹ : DecidableEq ι
      inst✝ : GradedAlgebra 𝒜
      e : ι
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      n : ι
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) j)
      ⊢ Eq (HomogeneousLocalization.mk { deg := HAdd.hAdd n (HSMul.hSMul j e), num : …
    -/
  · simp [awayMapAux_mk 𝒜 (hx := hx)]
    /-
      🎉 no goals
    -/
    /-
      case w.refine_1
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      𝒜 : ι → Submodule R A
      inst✝² : AddCommMonoid ι
      inst✝¹ : DecidableEq ι
      inst✝ : GradedAlgebra 𝒜
      e : ι
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      n : ι
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) j)
      ⊢ Membership.mem (𝒜 (HAdd.hAdd n (HSMul.hSMul j e))) (HMul.hMul a (HPow.hPow g …
    -/
  · apply SetLike.mul_mem_graded ha
    /-
      case w.refine_1
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      𝒜 : ι → Submodule R A
      inst✝² : AddCommMonoid ι
      inst✝¹ : DecidableEq ι
      inst✝ : GradedAlgebra 𝒜
      e : ι
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      n : ι
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) j)
      ⊢ Membership.mem (𝒜 (HSMul.hSMul j e)) (HPow.hPow g j)
    -/
    exact SetLike.pow_mem_graded _ hg
    /-
      🎉 no goals
    -/
    /-
      case w.refine_2
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      𝒜 : ι → Submodule R A
      inst✝² : AddCommMonoid ι
      inst✝¹ : DecidableEq ι
      inst✝ : GradedAlgebra 𝒜
      e : ι
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      n : ι
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) j)
      ⊢ Membership.mem (𝒜 (HAdd.hAdd n (HSMul.hSMul j e))) (HPow.hPow x j)
    -/
  · rw [hx, mul_pow]
    /-
      case w.refine_2
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      𝒜 : ι → Submodule R A
      inst✝² : AddCommMonoid ι
      inst✝¹ : DecidableEq ι
      inst✝ : GradedAlgebra 𝒜
      e : ι
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      n : ι
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) j)
      ⊢ Membership.mem (𝒜 (HAdd.hAdd n (HSMul.hSMul j e))) (HMul.hMul (HPow.hPow f j …
    -/
    apply SetLike.mul_mem_graded hb'
    /-
      case w.refine_2
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      𝒜 : ι → Submodule R A
      inst✝² : AddCommMonoid ι
      inst✝¹ : DecidableEq ι
      inst✝ : GradedAlgebra 𝒜
      e : ι
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      n : ι
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x => HPow.hPow f x) j)
      ⊢ Membership.mem (𝒜 (HSMul.hSMul j e)) (HPow.hPow g j)
    -/
    exact SetLike.pow_mem_graded _ hg
    /-
      🎉 no goals
    -/


/-- Given `x = f * g` with `g` homogeneous of positive degree,
this is the map `A_{(f)} → A_{(x)}` taking `a/f^i` to `ag^i/(fg)^i`. -/
def awayMap : Away 𝒜 f →+* Away 𝒜 x := by
  let e := RingEquiv.ofLeftInverse (f := algebraMap (Away 𝒜 x) (Localization.Away x))
    (h := (val_injective _).hasLeftInverse.choose_spec)
  refine RingHom.comp (e.symm.toRingHom.comp (Subring.inclusion ?_))
    (awayMapAux 𝒜 (f := f) ⟨_, hx⟩).rangeRestrict
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    x✝ : Submonoid A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e✝ : ι
    f g : A
    hg : Membership.mem (𝒜 e✝) g
    x : A
    hx : Eq x (HMul.hMul f g)
    e : RingEquiv (HomogeneousLocalization.Away 𝒜 x) (Subtype fun x_1 => Membershi …
    ⊢ LE.le (HomogeneousLocalization.awayMapAux 𝒜 ⋯).range (algebraMap (Homogeneou …
  -/
  exact range_awayMapAux_subset 𝒜 hg hx
  /-
    🎉 no goals
  -/


lemma val_awayMap_eq_aux (a) : (awayMap 𝒜 hg hx a).val = awayMapAux 𝒜 ⟨_, hx⟩ a := by
  let e := RingEquiv.ofLeftInverse (f := algebraMap (Away 𝒜 x) (Localization.Away x))
    (h := (val_injective _).hasLeftInverse.choose_spec)
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e✝ : ι
    f g : A
    hg : Membership.mem (𝒜 e✝) g
    x : A
    hx : Eq x (HMul.hMul f g)
    a : HomogeneousLocalization.Away 𝒜 f
    e : RingEquiv (HomogeneousLocalization.Away 𝒜 x) (Subtype fun x_1 => Membershi …
    ⊢ Eq (HomogeneousLocalization.val ((HomogeneousLocalization.awayMap 𝒜 hg hx) a …
  -/
  dsimp [awayMap]
  convert_to (e (e.symm ⟨awayMapAux 𝒜 (f := f) ⟨_, hx⟩ a,
    range_awayMapAux_subset 𝒜 hg hx ⟨_, rfl⟩⟩)).1 = _
  /-
    case convert_2
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e✝ : ι
    f g : A
    hg : Membership.mem (𝒜 e✝) g
    x : A
    hx : Eq x (HMul.hMul f g)
    a : HomogeneousLocalization.Away 𝒜 f
    e : RingEquiv (HomogeneousLocalization.Away 𝒜 x) (Subtype fun x_1 => Membershi …
    ⊢ Eq (↑(e (e.symm ⟨(HomogeneousLocalization.awayMapAux 𝒜 ⋯) a, ⋯⟩))) ((Homogen …
  -/
  rw [e.apply_symm_apply]
  /-
    🎉 no goals
  -/


lemma val_awayMap (a) : (awayMap 𝒜 hg hx a).val = Localization.awayLift (algebraMap A _) _
    (isUnit_of_dvd_unit (map_dvd _ ⟨_, hx⟩) (IsLocalization.Away.algebraMap_isUnit x)) a.val := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e : ι
    f g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    a : HomogeneousLocalization.Away 𝒜 f
    ⊢ Eq (HomogeneousLocalization.val ((HomogeneousLocalization.awayMap 𝒜 hg hx) a …
  -/
  rw [val_awayMap_eq_aux]
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e : ι
    f g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    a : HomogeneousLocalization.Away 𝒜 f
    ⊢ Eq ((HomogeneousLocalization.awayMapAux 𝒜 ⋯) a) ((Localization.awayLift (alg …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma awayMap_fromZeroRingHom (a) :
    awayMap 𝒜 hg hx (fromZeroRingHom 𝒜 _ a) = fromZeroRingHom 𝒜 _ a := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e : ι
    f g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    a : Subtype fun x => Membership.mem (𝒜 0) x
    ⊢ Eq ((HomogeneousLocalization.awayMap 𝒜 hg hx) ((HomogeneousLocalization.from …
  -/
  ext
  simp only [fromZeroRingHom, RingHom.coe_mk, MonoidHom.coe_mk, OneHom.coe_mk,
    val_awayMap, val_mk, SetLike.GradeZero.coe_one]
  /-
    case a
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e : ι
    f g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    a : Subtype fun x => Membership.mem (𝒜 0) x
    ⊢ Eq ((Localization.awayLift (algebraMap A (Localization (Submonoid.powers x)) …
  -/
  convert IsLocalization.lift_eq _ _
  /-
    🎉 no goals
  -/


lemma val_awayMap_mk (n a i hi) : (awayMap 𝒜 hg hx (mk ⟨n, a, ⟨f ^ i, hi⟩, ⟨i, rfl⟩⟩)).val =
    Localization.mk (a * g ^ i) ⟨x ^ i, (Submonoid.mem_powers_iff _ _).mpr ⟨i, rfl⟩⟩ := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e : ι
    f g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    n : ι
    a : Subtype fun x => Membership.mem (𝒜 n) x
    i : Nat
    hi : Membership.mem (𝒜 n) (HPow.hPow f i)
    ⊢ Eq (HomogeneousLocalization.val ((HomogeneousLocalization.awayMap 𝒜 hg hx) ( …
  -/
  rw [val_awayMap_eq_aux, awayMapAux_mk 𝒜 (hx := hx)]
  /-
    🎉 no goals
  -/


/-- Given `x = f * g` with `g` homogeneous of positive degree,
this is the map `A_{(f)} → A_{(x)}` taking `a/f^i` to `ag^i/(fg)^i`. -/
def awayMapₐ : Away 𝒜 f →ₐ[𝒜 0] Away 𝒜 x where
  __ := awayMap 𝒜 hg hx
  commutes' _ := awayMap_fromZeroRingHom ..


@[simp] lemma awayMapₐ_apply (a) : awayMapₐ 𝒜 hg hx a = awayMap 𝒜 hg hx a := rfl


/-- This is a convenient constructor for `Away 𝒜 f` when `f` is homogeneous.
`Away.mk 𝒜 hf n x hx` is the fraction `x / f ^ n`. -/
protected def Away.mk {d : ι} (hf : f ∈ 𝒜 d) (n : ℕ) (x : A) (hx : x ∈ 𝒜 (n • d)) : Away 𝒜 f :=
  .mk ⟨n • d, ⟨x, hx⟩, ⟨f ^ n, SetLike.pow_mem_graded n hf⟩, ⟨n, rfl⟩⟩


@[simp]
lemma Away.val_mk {d : ι} (n : ℕ) (hf : f ∈ 𝒜 d) (x : A) (hx : x ∈ 𝒜 (n • d)) :
                                                             /-
                                                               ι : Type u_1
                                                               R : Type u_2
                                                               A : Type u_3
                                                               inst✝⁵ : CommRing R
                                                               inst✝⁴ : CommRing A
                                                               inst✝³ : Algebra R A
                                                               𝒜 : ι → Submodule R A
                                                               x✝¹ : Submonoid A
                                                               inst✝² : AddCommMonoid ι
                                                               inst✝¹ : DecidableEq ι
                                                               inst✝ : GradedAlgebra 𝒜
                                                               e : ι
                                                               f g : A
                                                               hg : Membership.mem (𝒜 e) g
                                                               x✝ : A
                                                               hx✝ : Eq x✝ (HMul.hMul f g)
                                                               d : ι
                                                               n : Nat
                                                               hf : Membership.mem (𝒜 d) f
                                                               x : A
                                                               hx : Membership.mem (𝒜 (HSMul.hSMul n d)) x
                                                               ⊢ Membership.mem (Submonoid.powers f) (HPow.hPow f n)
                                                             -/
    (Away.mk 𝒜 hf n x hx).val = Localization.mk x ⟨f ^ n, by use n⟩ :=
                                                             /-
                                                               🎉 no goals
                                                             -/
  rfl


protected
lemma Away.mk_surjective {d : ι} (hf : f ∈ 𝒜 d) (x : Away 𝒜 f) :
    ∃ n a ha, Away.mk 𝒜 hf n a ha = x := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    f : A
    d : ι
    hf : Membership.mem (𝒜 d) f
    x : HomogeneousLocalization.Away 𝒜 f
    ⊢ Exists fun n => Exists fun a => Exists fun ha => Eq (HomogeneousLocalization …
  -/
  obtain ⟨⟨N, ⟨s, hs⟩, ⟨b, hn⟩, ⟨n, (rfl : _ = b)⟩⟩, rfl⟩ := mk_surjective x
  /-
    case intro.mk.mk.mk.intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    f : A
    d : ι
    hf : Membership.mem (𝒜 d) f
    N : ι
    s : A
    hs : Membership.mem (𝒜 N) s
    n : Nat
    hn : Membership.mem (𝒜 N) ((fun x => HPow.hPow f x) n)
    ⊢ Exists fun n_1 => Exists fun a => Exists fun ha => Eq (HomogeneousLocalizati …
  -/
  by_cases hfn : f ^ n = 0
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      𝒜 : ι → Submodule R A
      inst✝² : AddCommMonoid ι
      inst✝¹ : DecidableEq ι
      inst✝ : GradedAlgebra 𝒜
      f : A
      d : ι
      hf : Membership.mem (𝒜 d) f
      N : ι
      s : A
      hs : Membership.mem (𝒜 N) s
      n : Nat
      hn : Membership.mem (𝒜 N) ((fun x => HPow.hPow f x) n)
      hfn : Eq (HPow.hPow f n) 0
      ⊢ Exists fun n_1 => Exists fun a => Exists fun ha => Eq (HomogeneousLocalizati …
    -/
  · have := HomogeneousLocalization.subsingleton 𝒜 (x := .powers f) ⟨n, hfn⟩
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      𝒜 : ι → Submodule R A
      inst✝² : AddCommMonoid ι
      inst✝¹ : DecidableEq ι
      inst✝ : GradedAlgebra 𝒜
      f : A
      d : ι
      hf : Membership.mem (𝒜 d) f
      N : ι
      s : A
      hs : Membership.mem (𝒜 N) s
      n : Nat
      hn : Membership.mem (𝒜 N) ((fun x => HPow.hPow f x) n)
      hfn : Eq (HPow.hPow f n) 0
      this : Subsingleton (HomogeneousLocalization 𝒜 (Submonoid.powers f))
      ⊢ Exists fun n_1 => Exists fun a => Exists fun ha => Eq (HomogeneousLocalizati …
    -/
    exact ⟨0, 0, zero_mem _, Subsingleton.elim _ _⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    f : A
    d : ι
    hf : Membership.mem (𝒜 d) f
    N : ι
    s : A
    hs : Membership.mem (𝒜 N) s
    n : Nat
    hn : Membership.mem (𝒜 N) ((fun x => HPow.hPow f x) n)
    hfn : Not (Eq (HPow.hPow f n) 0)
    ⊢ Exists fun n_1 => Exists fun a => Exists fun ha => Eq (HomogeneousLocalizati …
  -/
  obtain rfl := DirectSum.degree_eq_of_mem_mem 𝒜 hn (SetLike.pow_mem_graded n hf) hfn
  /-
    case neg
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    f : A
    d : ι
    hf : Membership.mem (𝒜 d) f
    s : A
    n : Nat
    hfn : Not (Eq (HPow.hPow f n) 0)
    hs : Membership.mem (𝒜 (HSMul.hSMul n d)) s
    hn : Membership.mem (𝒜 (HSMul.hSMul n d)) ((fun x => HPow.hPow f x) n)
    ⊢ Exists fun n_1 => Exists fun a => Exists fun ha => Eq (HomogeneousLocalizati …
  -/
  exact ⟨n, s, hs, by ext; simp⟩
  /-
    🎉 no goals
  -/


open SetLike in
@[simp]
lemma awayMap_mk {d : ι} (n : ℕ) (hf : f ∈ 𝒜 d) (a : A) (ha : a ∈ 𝒜 (n • d)) :
    awayMap 𝒜 hg hx (Away.mk 𝒜 hf n a ha) = Away.mk 𝒜 (hx ▸ mul_mem_graded hf hg) n
                      /-
                        ι : Type u_1
                        R : Type u_2
                        A : Type u_3
                        inst✝⁵ : CommRing R
                        inst✝⁴ : CommRing A
                        inst✝³ : Algebra R A
                        𝒜 : ι → Submodule R A
                        x✝ : Submonoid A
                        inst✝² : AddCommMonoid ι
                        inst✝¹ : DecidableEq ι
                        inst✝ : GradedAlgebra 𝒜
                        e : ι
                        f g : A
                        hg : Membership.mem (𝒜 e) g
                        x : A
                        hx : Eq x (HMul.hMul f g)
                        d : ι
                        n : Nat
                        hf : Membership.mem (𝒜 d) f
                        a : A
                        ha : Membership.mem (𝒜 (HSMul.hSMul n d)) a
                        ⊢ Membership.mem (𝒜 (HSMul.hSMul n (HAdd.hAdd d e))) (HMul.hMul a (HPow.hPow g …
                      -/
      (a * g ^ n) (by rw [smul_add]; exact mul_mem_graded ha (pow_mem_graded n hg)) := by
                                     /-
                                       🎉 no goals
                                     -/
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e : ι
    f g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    d : ι
    n : Nat
    hf : Membership.mem (𝒜 d) f
    a : A
    ha : Membership.mem (𝒜 (HSMul.hSMul n d)) a
    ⊢ Eq ((HomogeneousLocalization.awayMap 𝒜 hg hx) (HomogeneousLocalization.Away. …
  -/
  ext
  /-
    case a
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    𝒜 : ι → Submodule R A
    inst✝² : AddCommMonoid ι
    inst✝¹ : DecidableEq ι
    inst✝ : GradedAlgebra 𝒜
    e : ι
    f g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    d : ι
    n : Nat
    hf : Membership.mem (𝒜 d) f
    a : A
    ha : Membership.mem (𝒜 (HSMul.hSMul n d)) a
    ⊢ Eq (HomogeneousLocalization.val ((HomogeneousLocalization.awayMap 𝒜 hg hx) ( …
  -/
  exact val_awayMap_mk ..
  /-
    🎉 no goals
  -/


/-- The element `t := g ^ d / f ^ e` such that `A_{(fg)} = A_{(f)}[1/t]`. -/
abbrev Away.isLocalizationElem : Away 𝒜 f :=
                             /-
                               ι : Type u_1
                               R : Type u_2
                               A : Type u_3
                               inst✝³ : CommRing R
                               inst✝² : CommRing A
                               inst✝¹ : Algebra R A
                               𝒜✝ : ι → Submodule R A
                               x : Submonoid A
                               𝒜 : Nat → Submodule R A
                               inst✝ : GradedAlgebra 𝒜
                               e d : Nat
                               f : A
                               hf : Membership.mem (𝒜 d) f
                               g : A
                               hg : Membership.mem (𝒜 e) g
                               ⊢ Membership.mem (𝒜 (HSMul.hSMul e d)) (HPow.hPow g d)
                             -/
  Away.mk 𝒜 hf e (g ^ d) (by convert SetLike.pow_mem_graded d hg using 2; exact mul_comm _ _)
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- Let `t := g ^ d / f ^ e`, then `A_{(fg)} = A_{(f)}[1/t]`. -/
theorem Away.isLocalization_mul (hd : d ≠ 0) :
    letI := (awayMap 𝒜 hg hx).toAlgebra
    IsLocalization.Away (isLocalizationElem hf hg) (Away 𝒜 x) := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    e d : Nat
    f : A
    hf : Membership.mem (𝒜 d) f
    g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    hd : Ne d 0
    ⊢ IsLocalization.Away (HomogeneousLocalization.Away.isLocalizationElem hf hg)  …
  -/
  letI := (awayMap 𝒜 hg hx).toAlgebra
  /-
    R : Type u_2
    A : Type u_3
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    e d : Nat
    f : A
    hf : Membership.mem (𝒜 d) f
    g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    hd : Ne d 0
    this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
    ⊢ IsLocalization.Away (HomogeneousLocalization.Away.isLocalizationElem hf hg)  …
  -/
  constructor
    /-
      case map_units'
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      ⊢ ∀ (y : Subtype fun x => Membership.mem (Submonoid.powers (HomogeneousLocaliz …
    -/
  · rintro ⟨r, n, rfl⟩
    /-
      case map_units'.mk.intro
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      ⊢ IsUnit ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalizat …
    -/
    rw [map_pow, RingHom.algebraMap_toAlgebra]
    let z : Away 𝒜 x := Away.mk 𝒜 (hx ▸ SetLike.mul_mem_graded hf hg) (d + e)
        (g ^ e * f ^ (2 * e + d)) <| by
      convert SetLike.mul_mem_graded (SetLike.pow_mem_graded e hg)
        (SetLike.pow_mem_graded (2 * e + d) hf) using 2
      ring
    /-
      case map_units'.mk.intro
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      z : HomogeneousLocalization.Away 𝒜 x := HomogeneousLocalization.Away.mk 𝒜 ⋯ (H …
      ⊢ IsUnit (HPow.hPow ((HomogeneousLocalization.awayMap 𝒜 hg hx) (HomogeneousLoc …
    -/
    refine (isUnit_iff_exists_inv.mpr ⟨z, ?_⟩).pow _
    /-
      case map_units'.mk.intro
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      z : HomogeneousLocalization.Away 𝒜 x := HomogeneousLocalization.Away.mk 𝒜 ⋯ (H …
      ⊢ Eq (HMul.hMul ((HomogeneousLocalization.awayMap 𝒜 hg hx) (HomogeneousLocaliz …
    -/
    ext
    /-
      case map_units'.mk.intro.a
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      z : HomogeneousLocalization.Away 𝒜 x := HomogeneousLocalization.Away.mk 𝒜 ⋯ (H …
      ⊢ Eq (HomogeneousLocalization.val (HMul.hMul ((HomogeneousLocalization.awayMap …
    -/
    simp only [val_mul, val_one, awayMap_mk, Away.val_mk, z, Localization.mk_mul]
    /-
      case map_units'.mk.intro.a
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      z : HomogeneousLocalization.Away 𝒜 x := HomogeneousLocalization.Away.mk 𝒜 ⋯ (H …
      ⊢ Eq (Localization.mk (HMul.hMul (HMul.hMul (HPow.hPow g d) (HPow.hPow g e)) ( …
    -/
    rw [← Localization.mk_one, Localization.mk_eq_mk_iff, Localization.r_iff_exists]
    /-
      case map_units'.mk.intro.a
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      z : HomogeneousLocalization.Away 𝒜 x := HomogeneousLocalization.Away.mk 𝒜 ⋯ (H …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul ↑{ fst := 1, snd := 1 }.2 { fs …
    -/
    use 1
    /-
      case h
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      z : HomogeneousLocalization.Away 𝒜 x := HomogeneousLocalization.Away.mk 𝒜 ⋯ (H …
      ⊢ Eq (HMul.hMul (↑1) (HMul.hMul ↑{ fst := 1, snd := 1 }.2 { fst := HMul.hMul ( …
    -/
    simp only [OneMemClass.coe_one, one_mul, Submonoid.coe_mul, mul_one, hx]
    /-
      case h
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      z : HomogeneousLocalization.Away 𝒜 x := HomogeneousLocalization.Away.mk 𝒜 ⋯ (H …
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow g d) (HPow.hPow g e)) (HMul.hMul (HPow.h …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case surj'
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      ⊢ ∀ (z : HomogeneousLocalization.Away 𝒜 x), Exists fun x_1 => Eq (HMul.hMul z  …
    -/
  · intro z
    /-
      case surj'
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      z : HomogeneousLocalization.Away 𝒜 x
      ⊢ Exists fun x_1 => Eq (HMul.hMul z ((algebraMap (HomogeneousLocalization.Away …
    -/
    obtain ⟨n, s, hs, rfl⟩ := Away.mk_surjective 𝒜 (hx ▸ SetLike.mul_mem_graded hf hg) z
    /-
      case surj'.intro.intro.intro
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      s : A
      hs : Membership.mem (𝒜 (HSMul.hSMul n (HAdd.hAdd d e))) s
      ⊢ Exists fun x_1 => Eq (HMul.hMul (HomogeneousLocalization.Away.mk 𝒜 ⋯ n s hs) …
    -/
    cases' d with d
      /-
        case surj'.intro.intro.intro.zero
        R : Type u_2
        A : Type u_3
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        e : Nat
        f g : A
        hg : Membership.mem (𝒜 e) g
        x : A
        hx : Eq x (HMul.hMul f g)
        this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
        n : Nat
        s : A
        hf : Membership.mem (𝒜 0) f
        hd : Ne 0 0
        hs : Membership.mem (𝒜 (HSMul.hSMul n (HAdd.hAdd 0 e))) s
        ⊢ Exists fun x_1 => Eq (HMul.hMul (HomogeneousLocalization.Away.mk 𝒜 ⋯ n s hs) …
      -/
    · contradiction
      /-
        🎉 no goals
      -/
    let t : Away 𝒜 f := Away.mk 𝒜 hf (n * (e + 1)) (s * g ^ (n * d)) <| by
      convert SetLike.mul_mem_graded hs (SetLike.pow_mem_graded _ hg) using 2; simp; ring
    /-
      case surj'.intro.intro.intro.succ
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e : Nat
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      s : A
      d : Nat
      hf : Membership.mem (𝒜 (HAdd.hAdd d 1)) f
      hd : Ne (HAdd.hAdd d 1) 0
      hs : Membership.mem (𝒜 (HSMul.hSMul n (HAdd.hAdd (HAdd.hAdd d 1) e))) s
      t : HomogeneousLocalization.Away 𝒜 f := HomogeneousLocalization.Away.mk 𝒜 hf ( …
      ⊢ Exists fun x_1 => Eq (HMul.hMul (HomogeneousLocalization.Away.mk 𝒜 ⋯ n s hs) …
    -/
    refine ⟨⟨t, ⟨_, ⟨n, rfl⟩⟩⟩, ?_⟩
    /-
      case surj'.intro.intro.intro.succ
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e : Nat
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      s : A
      d : Nat
      hf : Membership.mem (𝒜 (HAdd.hAdd d 1)) f
      hd : Ne (HAdd.hAdd d 1) 0
      hs : Membership.mem (𝒜 (HSMul.hSMul n (HAdd.hAdd (HAdd.hAdd d 1) e))) s
      t : HomogeneousLocalization.Away 𝒜 f := HomogeneousLocalization.Away.mk 𝒜 hf ( …
      ⊢ Eq (HMul.hMul (HomogeneousLocalization.Away.mk 𝒜 ⋯ n s hs) ((algebraMap (Hom …
    -/
    ext
    simp only [RingHom.algebraMap_toAlgebra, map_pow, awayMap_mk, val_mul, val_mk, val_pow,
      Localization.mk_pow, Localization.mk_mul, t]
    /-
      case surj'.intro.intro.intro.succ.a
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e : Nat
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      s : A
      d : Nat
      hf : Membership.mem (𝒜 (HAdd.hAdd d 1)) f
      hd : Ne (HAdd.hAdd d 1) 0
      hs : Membership.mem (𝒜 (HSMul.hSMul n (HAdd.hAdd (HAdd.hAdd d 1) e))) s
      t : HomogeneousLocalization.Away 𝒜 f := HomogeneousLocalization.Away.mk 𝒜 hf ( …
      ⊢ Eq (Localization.mk (HMul.hMul s (HPow.hPow (HMul.hMul (HPow.hPow g (HAdd.hA …
    -/
    rw [Localization.mk_eq_mk_iff, Localization.r_iff_exists]
    /-
      case surj'.intro.intro.intro.succ.a
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e : Nat
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      s : A
      d : Nat
      hf : Membership.mem (𝒜 (HAdd.hAdd d 1)) f
      hd : Ne (HAdd.hAdd d 1) 0
      hs : Membership.mem (𝒜 (HSMul.hSMul n (HAdd.hAdd (HAdd.hAdd d 1) e))) s
      t : HomogeneousLocalization.Away 𝒜 f := HomogeneousLocalization.Away.mk 𝒜 hf ( …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul ↑{ fst := HMul.hMul (HMul.hMul …
    -/
    exact ⟨1, by simp; ring⟩
    /-
      🎉 no goals
    -/
    /-
      case exists_of_eq
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      ⊢ ∀ {x_1 y : HomogeneousLocalization.Away 𝒜 f}, Eq ((algebraMap (HomogeneousLo …
    -/
  · intro a b e
    /-
      case exists_of_eq
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e✝ d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e✝) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      a b : HomogeneousLocalization.Away 𝒜 f
      e : Eq ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalizatio …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) a) (HMul.hMul (↑c) b)
    -/
    obtain ⟨n, a, ha, rfl⟩ := Away.mk_surjective 𝒜 hf a
    /-
      case exists_of_eq.intro.intro.intro
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e✝ d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e✝) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      b : HomogeneousLocalization.Away 𝒜 f
      n : Nat
      a : A
      ha : Membership.mem (𝒜 (HSMul.hSMul n d)) a
      e : Eq ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalizatio …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HomogeneousLocalization.Away.mk 𝒜 hf n a …
    -/
    obtain ⟨m, b, hb, rfl⟩ := Away.mk_surjective 𝒜 hf b
    /-
      case exists_of_eq.intro.intro.intro.intro.intro.intro
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e✝ d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e✝) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      a : A
      ha : Membership.mem (𝒜 (HSMul.hSMul n d)) a
      m : Nat
      b : A
      hb : Membership.mem (𝒜 (HSMul.hSMul m d)) b
      e : Eq ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalizatio …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HomogeneousLocalization.Away.mk 𝒜 hf n a …
    -/
    replace e := congr_arg val e
    simp only [RingHom.algebraMap_toAlgebra, awayMap_mk, val_mk,
      Localization.mk_eq_mk_iff, Localization.r_iff_exists] at e
    /-
      case exists_of_eq.intro.intro.intro.intro.intro.intro
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e✝ d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e✝) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      a : A
      ha : Membership.mem (𝒜 (HSMul.hSMul n d)) a
      m : Nat
      b : A
      hb : Membership.mem (𝒜 (HSMul.hSMul m d)) b
      e : Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul (HPow.hPow x m) (HMul.hMul a …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HomogeneousLocalization.Away.mk 𝒜 hf n a …
    -/
    obtain ⟨⟨_, k, rfl⟩, hc⟩ := e
    /-
      case exists_of_eq.intro.intro.intro.intro.intro.intro.intro.mk.intro
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      a : A
      ha : Membership.mem (𝒜 (HSMul.hSMul n d)) a
      m : Nat
      b : A
      hb : Membership.mem (𝒜 (HSMul.hSMul m d)) b
      k : Nat
      hc : Eq (HMul.hMul (↑⟨(fun x_1 => HPow.hPow x x_1) k, ⋯⟩) (HMul.hMul (HPow.hPo …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HomogeneousLocalization.Away.mk 𝒜 hf n a …
    -/
    refine ⟨⟨_, k + m + n, rfl⟩, ?_⟩
    /-
      case exists_of_eq.intro.intro.intro.intro.intro.intro.intro.mk.intro
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      a : A
      ha : Membership.mem (𝒜 (HSMul.hSMul n d)) a
      m : Nat
      b : A
      hb : Membership.mem (𝒜 (HSMul.hSMul m d)) b
      k : Nat
      hc : Eq (HMul.hMul (↑⟨(fun x_1 => HPow.hPow x x_1) k, ⋯⟩) (HMul.hMul (HPow.hPo …
      ⊢ Eq (HMul.hMul (↑⟨(fun x => HPow.hPow (HomogeneousLocalization.Away.isLocaliz …
    -/
    ext
    simp only [OneMemClass.coe_one, one_mul, val_mul, val_pow, val_mk, Localization.mk_pow,
      Localization.mk_eq_mk_iff, Localization.r_iff_exists, Submonoid.coe_mul, Localization.mk_mul,
      SubmonoidClass.coe_pow, Subtype.exists, exists_prop]
    /-
      case exists_of_eq.intro.intro.intro.intro.intro.intro.intro.mk.intro.a
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      a : A
      ha : Membership.mem (𝒜 (HSMul.hSMul n d)) a
      m : Nat
      b : A
      hb : Membership.mem (𝒜 (HSMul.hSMul m d)) b
      k : Nat
      hc : Eq (HMul.hMul (↑⟨(fun x_1 => HPow.hPow x x_1) k, ⋯⟩) (HMul.hMul (HPow.hPo …
      ⊢ Exists fun a_1 => And (Membership.mem (Submonoid.powers f) a_1) (Eq (HMul.hM …
    -/
    refine ⟨_, ⟨k, rfl⟩, ?_⟩
    /-
      case exists_of_eq.intro.intro.intro.intro.intro.intro.intro.mk.intro.a
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e d : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : Ne d 0
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      a : A
      ha : Membership.mem (𝒜 (HSMul.hSMul n d)) a
      m : Nat
      b : A
      hb : Membership.mem (𝒜 (HSMul.hSMul m d)) b
      k : Nat
      hc : Eq (HMul.hMul (↑⟨(fun x_1 => HPow.hPow x x_1) k, ⋯⟩) (HMul.hMul (HPow.hPo …
      ⊢ Eq (HMul.hMul ((fun x => HPow.hPow f x) k) (HMul.hMul (HMul.hMul (HPow.hPow  …
    -/
    cases' d with d
      /-
        case exists_of_eq.intro.intro.intro.intro.intro.intro.intro.mk.intro.a.zero
        R : Type u_2
        A : Type u_3
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        e : Nat
        f g : A
        hg : Membership.mem (𝒜 e) g
        x : A
        hx : Eq x (HMul.hMul f g)
        this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
        n : Nat
        a : A
        m : Nat
        b : A
        k : Nat
        hc : Eq (HMul.hMul (↑⟨(fun x_1 => HPow.hPow x x_1) k, ⋯⟩) (HMul.hMul (HPow.hPo …
        hf : Membership.mem (𝒜 0) f
        hd : Ne 0 0
        ha : Membership.mem (𝒜 (HSMul.hSMul n 0)) a
        hb : Membership.mem (𝒜 (HSMul.hSMul m 0)) b
        ⊢ Eq (HMul.hMul ((fun x => HPow.hPow f x) k) (HMul.hMul (HMul.hMul (HPow.hPow  …
      -/
    · contradiction
      /-
        🎉 no goals
      -/
    /-
      case exists_of_eq.intro.intro.intro.intro.intro.intro.intro.mk.intro.a.succ
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e : Nat
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      n : Nat
      a : A
      m : Nat
      b : A
      k : Nat
      hc : Eq (HMul.hMul (↑⟨(fun x_1 => HPow.hPow x x_1) k, ⋯⟩) (HMul.hMul (HPow.hPo …
      d : Nat
      hf : Membership.mem (𝒜 (HAdd.hAdd d 1)) f
      hd : Ne (HAdd.hAdd d 1) 0
      ha : Membership.mem (𝒜 (HSMul.hSMul n (HAdd.hAdd d 1))) a
      hb : Membership.mem (𝒜 (HSMul.hSMul m (HAdd.hAdd d 1))) b
      ⊢ Eq (HMul.hMul ((fun x => HPow.hPow f x) k) (HMul.hMul (HMul.hMul (HPow.hPow  …
    -/
    subst hx
    /-
      case exists_of_eq.intro.intro.intro.intro.intro.intro.intro.mk.intro.a.succ
      R : Type u_2
      A : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e : Nat
      f g : A
      hg : Membership.mem (𝒜 e) g
      n : Nat
      a : A
      m : Nat
      b : A
      k d : Nat
      hf : Membership.mem (𝒜 (HAdd.hAdd d 1)) f
      hd : Ne (HAdd.hAdd d 1) 0
      ha : Membership.mem (𝒜 (HSMul.hSMul n (HAdd.hAdd d 1))) a
      hb : Membership.mem (𝒜 (HSMul.hSMul m (HAdd.hAdd d 1))) b
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.Awa …
      hc : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow (HMul.hMul f g) x) k, ⋯⟩) (HMul.hMul …
      ⊢ Eq (HMul.hMul ((fun x => HPow.hPow f x) k) (HMul.hMul (HMul.hMul (HPow.hPow  …
    -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
    convert congr(f ^ (e * (k + m + n)) * g ^ (d * (k + m + n)) * $hc) using 1 <;> ring
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


