/-- Product of two distributive types is distributive. -/
instance instDistrib [Distrib R] [Distrib S] : Distrib (R × S) :=
  { left_distrib := fun _ _ _ => mk.inj_iff.mpr ⟨left_distrib _ _ _, left_distrib _ _ _⟩
    right_distrib := fun _ _ _ => mk.inj_iff.mpr ⟨right_distrib _ _ _, right_distrib _ _ _⟩ }


/-- Product of two `NonUnitalNonAssocSemiring`s is a `NonUnitalNonAssocSemiring`. -/
instance instNonUnitalNonAssocSemiring [NonUnitalNonAssocSemiring R] [NonUnitalNonAssocSemiring S] :
    NonUnitalNonAssocSemiring (R × S) :=
  { inferInstanceAs (AddCommMonoid (R × S)),
    inferInstanceAs (Distrib (R × S)),
    inferInstanceAs (MulZeroClass (R × S)) with }


/-- Product of two `NonUnitalSemiring`s is a `NonUnitalSemiring`. -/
instance instNonUnitalSemiring [NonUnitalSemiring R] [NonUnitalSemiring S] :
    NonUnitalSemiring (R × S) :=
  { inferInstanceAs (NonUnitalNonAssocSemiring (R × S)),
    inferInstanceAs (SemigroupWithZero (R × S)) with }


/-- Product of two `NonAssocSemiring`s is a `NonAssocSemiring`. -/
instance instNonAssocSemiring [NonAssocSemiring R] [NonAssocSemiring S] :
    NonAssocSemiring (R × S) :=
  { inferInstanceAs (NonUnitalNonAssocSemiring (R × S)),
    inferInstanceAs (MulZeroOneClass (R × S)),
    inferInstanceAs (AddMonoidWithOne (R × S)) with }


/-- Product of two semirings is a semiring. -/
instance instSemiring [Semiring R] [Semiring S] : Semiring (R × S) :=
  { inferInstanceAs (NonUnitalSemiring (R × S)),
    inferInstanceAs (NonAssocSemiring (R × S)),
    inferInstanceAs (MonoidWithZero (R × S)) with }


/-- Product of two `NonUnitalCommSemiring`s is a `NonUnitalCommSemiring`. -/
instance instNonUnitalCommSemiring [NonUnitalCommSemiring R] [NonUnitalCommSemiring S] :
    NonUnitalCommSemiring (R × S) :=
  { inferInstanceAs (NonUnitalSemiring (R × S)), inferInstanceAs (CommSemigroup (R × S)) with }


/-- Product of two commutative semirings is a commutative semiring. -/
instance instCommSemiring [CommSemiring R] [CommSemiring S] : CommSemiring (R × S) :=
  { inferInstanceAs (Semiring (R × S)), inferInstanceAs (CommMonoid (R × S)) with }


instance instNonUnitalNonAssocRing [NonUnitalNonAssocRing R] [NonUnitalNonAssocRing S] :
    NonUnitalNonAssocRing (R × S) :=
  { inferInstanceAs (AddCommGroup (R × S)),
    inferInstanceAs (NonUnitalNonAssocSemiring (R × S)) with }


instance instNonUnitalRing [NonUnitalRing R] [NonUnitalRing S] : NonUnitalRing (R × S) :=
  { inferInstanceAs (NonUnitalNonAssocRing (R × S)),
    inferInstanceAs (NonUnitalSemiring (R × S)) with }


instance instNonAssocRing [NonAssocRing R] [NonAssocRing S] : NonAssocRing (R × S) :=
  { inferInstanceAs (NonUnitalNonAssocRing (R × S)),
    inferInstanceAs (NonAssocSemiring (R × S)),
    inferInstanceAs (AddGroupWithOne (R × S)) with }


/-- Product of two rings is a ring. -/
instance instRing [Ring R] [Ring S] : Ring (R × S) :=
  { inferInstanceAs (Semiring (R × S)),
    inferInstanceAs (AddCommGroup (R × S)),
    inferInstanceAs (AddGroupWithOne (R × S)) with }


/-- Product of two `NonUnitalCommRing`s is a `NonUnitalCommRing`. -/
instance instNonUnitalCommRing [NonUnitalCommRing R] [NonUnitalCommRing S] :
    NonUnitalCommRing (R × S) :=
  { inferInstanceAs (NonUnitalRing (R × S)), inferInstanceAs (CommSemigroup (R × S)) with }


/-- Product of two commutative rings is a commutative ring. -/
instance instCommRing [CommRing R] [CommRing S] : CommRing (R × S) :=
  { inferInstanceAs (Ring (R × S)), inferInstanceAs (CommMonoid (R × S)) with }


/-- Given non-unital semirings `R`, `S`, the natural projection homomorphism from `R × S` to `R`. -/
def fst : R × S →ₙ+* R :=
  { MulHom.fst R S, AddMonoidHom.fst R S with toFun := Prod.fst }


/-- Given non-unital semirings `R`, `S`, the natural projection homomorphism from `R × S` to `S`. -/
def snd : R × S →ₙ+* S :=
  { MulHom.snd R S, AddMonoidHom.snd R S with toFun := Prod.snd }


@[simp]
theorem coe_fst : ⇑(fst R S) = Prod.fst :=
  rfl


@[simp]
theorem coe_snd : ⇑(snd R S) = Prod.snd :=
  rfl


/-- Combine two non-unital ring homomorphisms `f : R →ₙ+* S`, `g : R →ₙ+* T` into
`f.prod g : R →ₙ+* S × T` given by `(f.prod g) x = (f x, g x)` -/
protected def prod (f : R →ₙ+* S) (g : R →ₙ+* T) : R →ₙ+* S × T :=
  { MulHom.prod (f : MulHom R S) (g : MulHom R T), AddMonoidHom.prod (f : R →+ S) (g : R →+ T) with
    toFun := fun x => (f x, g x) }


@[simp]
theorem prod_apply (x) : f.prod g x = (f x, g x) :=
  rfl


@[simp]
theorem fst_comp_prod : (fst S T).comp (f.prod g) = f :=
  ext fun _ => rfl


@[simp]
theorem snd_comp_prod : (snd S T).comp (f.prod g) = g :=
  ext fun _ => rfl


theorem prod_unique (f : R →ₙ+* S × T) : ((fst S T).comp f).prod ((snd S T).comp f) = f :=
                  /-
                    R : Type u_1
                    S : Type u_3
                    T : Type u_5
                    inst✝² : NonUnitalNonAssocSemiring R
                    inst✝¹ : NonUnitalNonAssocSemiring S
                    inst✝ : NonUnitalNonAssocSemiring T
                    f : NonUnitalRingHom R (Prod S T)
                    x : R
                    ⊢ Eq ((((NonUnitalRingHom.fst S T).comp f).prod ((NonUnitalRingHom.snd S T).co …
                  -/
  ext fun x => by simp only [prod_apply, coe_fst, coe_snd, comp_apply]
                  /-
                    🎉 no goals
                  -/


/-- `Prod.map` as a `NonUnitalRingHom`. -/
def prodMap : R × S →ₙ+* R' × S' :=
  (f.comp (fst R S)).prod (g.comp (snd R S))


theorem prodMap_def : prodMap f g = (f.comp (fst R S)).prod (g.comp (snd R S)) :=
  rfl


@[simp]
theorem coe_prodMap : ⇑(prodMap f g) = Prod.map f g :=
  rfl


theorem prod_comp_prodMap (f : T →ₙ+* R) (g : T →ₙ+* S) (f' : R →ₙ+* R') (g' : S →ₙ+* S') :
    (f'.prodMap g').comp (f.prod g) = (f'.comp f).prod (g'.comp g) :=
  rfl


/-- Given semirings `R`, `S`, the natural projection homomorphism from `R × S` to `R`. -/
def fst : R × S →+* R :=
  { MonoidHom.fst R S, AddMonoidHom.fst R S with toFun := Prod.fst }


/-- Given semirings `R`, `S`, the natural projection homomorphism from `R × S` to `S`. -/
def snd : R × S →+* S :=
  { MonoidHom.snd R S, AddMonoidHom.snd R S with toFun := Prod.snd }


instance (R S) [Semiring R] [Semiring S] : RingHomSurjective (fst R S) := ⟨(⟨⟨·, 0⟩, rfl⟩)⟩

instance (R S) [Semiring R] [Semiring S] : RingHomSurjective (snd R S) := ⟨(⟨⟨0, ·⟩, rfl⟩)⟩


/-- Combine two ring homomorphisms `f : R →+* S`, `g : R →+* T` into `f.prod g : R →+* S × T`
given by `(f.prod g) x = (f x, g x)` -/
protected def prod (f : R →+* S) (g : R →+* T) : R →+* S × T :=
  { MonoidHom.prod (f : R →* S) (g : R →* T), AddMonoidHom.prod (f : R →+ S) (g : R →+ T) with
    toFun := fun x => (f x, g x) }


theorem prod_unique (f : R →+* S × T) : ((fst S T).comp f).prod ((snd S T).comp f) = f :=
                  /-
                    R : Type u_1
                    S : Type u_3
                    T : Type u_5
                    inst✝² : NonAssocSemiring R
                    inst✝¹ : NonAssocSemiring S
                    inst✝ : NonAssocSemiring T
                    f : RingHom R (Prod S T)
                    x : R
                    ⊢ Eq ((((RingHom.fst S T).comp f).prod ((RingHom.snd S T).comp f)) x) (f x)
                  -/
  ext fun x => by simp only [prod_apply, coe_fst, coe_snd, comp_apply]
                  /-
                    🎉 no goals
                  -/


/-- `Prod.map` as a `RingHom`. -/
def prodMap : R × S →+* R' × S' :=
  (f.comp (fst R S)).prod (g.comp (snd R S))


theorem prod_comp_prodMap (f : T →+* R) (g : T →+* S) (f' : R →+* R') (g' : S →+* S') :
    (f'.prodMap g').comp (f.prod g) = (f'.comp f).prod (g'.comp g) :=
  rfl


/-- Swapping components as an equivalence of (semi)rings. -/
def prodComm : R × S ≃+* S × R :=
  { AddEquiv.prodComm, MulEquiv.prodComm with }


@[simp]
theorem coe_prodComm : ⇑(prodComm : R × S ≃+* S × R) = Prod.swap :=
  rfl


@[simp]
theorem coe_prodComm_symm : ⇑(prodComm : R × S ≃+* S × R).symm = Prod.swap :=
  rfl


@[simp]
theorem fst_comp_coe_prodComm :
    (RingHom.fst S R).comp ↑(prodComm : R × S ≃+* S × R) = RingHom.snd R S :=
  RingHom.ext fun _ => rfl


@[simp]
theorem snd_comp_coe_prodComm :
    (RingHom.snd S R).comp ↑(prodComm : R × S ≃+* S × R) = RingHom.fst R S :=
  RingHom.ext fun _ => rfl


/-- Four-way commutativity of `Prod`. The name matches `mul_mul_mul_comm`. -/
@[simps apply]
def prodProdProdComm : (R × R') × S × S' ≃+* (R × S) × R' × S' :=
  { AddEquiv.prodProdProdComm R R' S S', MulEquiv.prodProdProdComm R R' S S' with
    toFun := fun rrss => ((rrss.1.1, rrss.2.1), (rrss.1.2, rrss.2.2))
    invFun := fun rsrs => ((rsrs.1.1, rsrs.2.1), (rsrs.1.2, rsrs.2.2)) }


@[simp]
theorem prodProdProdComm_symm : (prodProdProdComm R R' S S').symm = prodProdProdComm R S R' S' :=
  rfl


@[simp]
theorem prodProdProdComm_toAddEquiv :
    (prodProdProdComm R R' S S' : _ ≃+ _) = AddEquiv.prodProdProdComm R R' S S' :=
  rfl


@[simp]
theorem prodProdProdComm_toMulEquiv :
    (prodProdProdComm R R' S S' : _ ≃* _) = MulEquiv.prodProdProdComm R R' S S' :=
  rfl


@[simp]
theorem prodProdProdComm_toEquiv :
    (prodProdProdComm R R' S S' : _ ≃ _) = Equiv.prodProdProdComm R R' S S' :=
  rfl


/-- A ring `R` is isomorphic to `R × S` when `S` is the zero ring -/
@[simps]
def prodZeroRing : R ≃+* R × S where
  toFun x := (x, 0)
  invFun := Prod.fst
                 /-
                   R : Type u_1
                   R' : Type u_2
                   S : Type u_3
                   S' : Type u_4
                   T : Type u_5
                   inst✝⁴ : NonAssocSemiring R
                   inst✝³ : NonAssocSemiring S
                   inst✝² : NonAssocSemiring R'
                   inst✝¹ : NonAssocSemiring S'
                   inst✝ : Subsingleton S
                   ⊢ ∀ (x y : R), Eq ({ toFun := fun x => { fst := x, snd := 0 }, invFun := Prod. …
                 -/
                 /-
                   R : Type u_1
                   R' : Type u_2
                   S : Type u_3
                   S' : Type u_4
                   T : Type u_5
                   inst✝⁴ : NonAssocSemiring R
                   inst✝³ : NonAssocSemiring S
                   inst✝² : NonAssocSemiring R'
                   inst✝¹ : NonAssocSemiring S'
                   inst✝ : Subsingleton S
                   ⊢ ∀ (x y : R), Eq ({ toFun := fun x => { fst := x, snd := 0 }, invFun := Prod. …
                 -/
  map_add' := by simp
                    /-
                      R : Type u_1
                      R' : Type u_2
                      S : Type u_3
                      S' : Type u_4
                      T : Type u_5
                      inst✝⁴ : NonAssocSemiring R
                      inst✝³ : NonAssocSemiring S
                      inst✝² : NonAssocSemiring R'
                      inst✝¹ : NonAssocSemiring S'
                      inst✝ : Subsingleton S
                      x : Prod R S
                      ⊢ Eq ((fun x => { fst := x, snd := 0 }) x.1) x
                    -/
                 /-
                   🎉 no goals
                 -/
                             /-
                               🎉 no goals
                             -/
                 /-
                   🎉 no goals
                 -/
  map_mul' := by simp
  left_inv _ := rfl
  right_inv x := by cases x; simp [eq_iff_true_of_subsingleton]


/-- A ring `R` is isomorphic to `S × R` when `S` is the zero ring -/
@[simps]
def zeroRingProd : R ≃+* S × R where
  toFun x := (0, x)
  invFun := Prod.snd
                 /-
                   R : Type u_1
                   R' : Type u_2
                   S : Type u_3
                   S' : Type u_4
                   T : Type u_5
                   inst✝⁴ : NonAssocSemiring R
                   inst✝³ : NonAssocSemiring S
                   inst✝² : NonAssocSemiring R'
                   inst✝¹ : NonAssocSemiring S'
                   inst✝ : Subsingleton S
                   ⊢ ∀ (x y : R), Eq ({ toFun := fun x => { fst := 0, snd := x }, invFun := Prod. …
                 -/
                 /-
                   R : Type u_1
                   R' : Type u_2
                   S : Type u_3
                   S' : Type u_4
                   T : Type u_5
                   inst✝⁴ : NonAssocSemiring R
                   inst✝³ : NonAssocSemiring S
                   inst✝² : NonAssocSemiring R'
                   inst✝¹ : NonAssocSemiring S'
                   inst✝ : Subsingleton S
                   ⊢ ∀ (x y : R), Eq ({ toFun := fun x => { fst := 0, snd := x }, invFun := Prod. …
                 -/
  map_add' := by simp
                    /-
                      R : Type u_1
                      R' : Type u_2
                      S : Type u_3
                      S' : Type u_4
                      T : Type u_5
                      inst✝⁴ : NonAssocSemiring R
                      inst✝³ : NonAssocSemiring S
                      inst✝² : NonAssocSemiring R'
                      inst✝¹ : NonAssocSemiring S'
                      inst✝ : Subsingleton S
                      x : Prod S R
                      ⊢ Eq ((fun x => { fst := 0, snd := x }) x.2) x
                    -/
                 /-
                   🎉 no goals
                 -/
                             /-
                               🎉 no goals
                             -/
                 /-
                   🎉 no goals
                 -/
  map_mul' := by simp
  left_inv _ := rfl
  right_inv x := by cases x; simp [eq_iff_true_of_subsingleton]


/-- The product of two nontrivial rings is not a domain -/
theorem false_of_nontrivial_of_product_domain (R S : Type*) [Ring R] [Ring S] [IsDomain (R × S)]
    [Nontrivial R] [Nontrivial S] : False := by
  have :=
    NoZeroDivisors.eq_zero_or_eq_zero_of_mul_eq_zero (show ((0 : R), (1 : S)) * (1, 0) = 0 by simp)
  /-
    R : Type u_6
    S : Type u_7
    inst✝⁴ : Ring R
    inst✝³ : Ring S
    inst✝² : IsDomain (Prod R S)
    inst✝¹ : Nontrivial R
    inst✝ : Nontrivial S
    this : Or (Eq { fst := 0, snd := 1 } 0) (Eq { fst := 1, snd := 0 } 0)
    ⊢ False
  -/
  rw [Prod.mk_eq_zero, Prod.mk_eq_zero] at this
  /-
    R : Type u_6
    S : Type u_7
    inst✝⁴ : Ring R
    inst✝³ : Ring S
    inst✝² : IsDomain (Prod R S)
    inst✝¹ : Nontrivial R
    inst✝ : Nontrivial S
    this : Or (And (Eq 0 0) (Eq 1 0)) (And (Eq 1 0) (Eq 0 0))
    ⊢ False
  -/
  rcases this with (⟨_, h⟩ | ⟨h, _⟩)
    /-
      case inl.intro
      R : Type u_6
      S : Type u_7
      inst✝⁴ : Ring R
      inst✝³ : Ring S
      inst✝² : IsDomain (Prod R S)
      inst✝¹ : Nontrivial R
      inst✝ : Nontrivial S
      left✝ : Eq 0 0
      h : Eq 1 0
      ⊢ False
    -/
  · exact zero_ne_one h.symm
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      R : Type u_6
      S : Type u_7
      inst✝⁴ : Ring R
      inst✝³ : Ring S
      inst✝² : IsDomain (Prod R S)
      inst✝¹ : Nontrivial R
      inst✝ : Nontrivial S
      h : Eq 1 0
      right✝ : Eq 0 0
      ⊢ False
    -/
  · exact zero_ne_one h.symm
    /-
      🎉 no goals
    -/

