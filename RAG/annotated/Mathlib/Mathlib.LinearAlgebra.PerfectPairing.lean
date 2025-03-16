/-- A perfect pairing of two (left) modules over a commutative ring. -/
structure PerfectPairing where
  toLin : M →ₗ[R] N →ₗ[R] R
  bijectiveLeft : Bijective toLin
  bijectiveRight : Bijective toLin.flip


/-- If the coefficients are a field, and one of the spaces is finite-dimensional, it is sufficient
to check only injectivity instead of bijectivity of the bilinear form. -/
def mkOfInjective {K V W : Type*}
    [Field K] [AddCommGroup V] [Module K V] [AddCommGroup W] [Module K W] [FiniteDimensional K V]
    (B : V →ₗ[K] W →ₗ[K] K)
    (h : Injective B)
    (h' : Injective B.flip) :
    PerfectPairing K V W where
  toLin := B
                          /-
                            R : Type u_1
                            M : Type u_2
                            N : Type u_3
                            inst✝¹⁰ : CommRing R
                            inst✝⁹ : AddCommGroup M
                            inst✝⁸ : Module R M
                            inst✝⁷ : AddCommGroup N
                            inst✝⁶ : Module R N
                            K : Type u_4
                            V : Type u_5
                            W : Type u_6
                            inst✝⁵ : Field K
                            inst✝⁴ : AddCommGroup V
                            inst✝³ : Module K V
                            inst✝² : AddCommGroup W
                            inst✝¹ : Module K W
                            inst✝ : FiniteDimensional K V
                            B : LinearMap (RingHom.id K) V (LinearMap (RingHom.id K) W K)
                            h : Function.Injective ⇑B
                            h' : Function.Injective ⇑B.flip
                            ⊢ Function.Surjective ⇑B
                          -/
  bijectiveLeft := ⟨h, by rwa [← B.flip_injective_iff₁]⟩
                          /-
                            🎉 no goals
                          -/
  bijectiveRight := ⟨h', by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝¹⁰ : CommRing R
      inst✝⁹ : AddCommGroup M
      inst✝⁸ : Module R M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R N
      K : Type u_4
      V : Type u_5
      W : Type u_6
      inst✝⁵ : Field K
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module K V
      inst✝² : AddCommGroup W
      inst✝¹ : Module K W
      inst✝ : FiniteDimensional K V
      B : LinearMap (RingHom.id K) V (LinearMap (RingHom.id K) W K)
      h : Function.Injective ⇑B
      h' : Function.Injective ⇑B.flip
      ⊢ Function.Surjective ⇑B.flip
    -/
    have : FiniteDimensional K W := FiniteDimensional.of_injective B.flip h'
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝¹⁰ : CommRing R
      inst✝⁹ : AddCommGroup M
      inst✝⁸ : Module R M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R N
      K : Type u_4
      V : Type u_5
      W : Type u_6
      inst✝⁵ : Field K
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module K V
      inst✝² : AddCommGroup W
      inst✝¹ : Module K W
      inst✝ : FiniteDimensional K V
      B : LinearMap (RingHom.id K) V (LinearMap (RingHom.id K) W K)
      h : Function.Injective ⇑B
      h' : Function.Injective ⇑B.flip
      this : FiniteDimensional K W
      ⊢ Function.Surjective ⇑B.flip
    -/
    rwa [← B.flip.flip_injective_iff₁, LinearMap.flip_flip]⟩
    /-
      🎉 no goals
    -/


/-- If the coefficients are a field, and one of the spaces is finite-dimensional, it is sufficient
to check only injectivity instead of bijectivity of the bilinear form. -/
def mkOfInjective' {K V W : Type*}
    [Field K] [AddCommGroup V] [Module K V] [AddCommGroup W] [Module K W] [FiniteDimensional K W]
    (B : V →ₗ[K] W →ₗ[K] K)
    (h : Injective B)
    (h' : Injective B.flip) :
    PerfectPairing K V W where
  toLin := B
  bijectiveLeft := ⟨h, by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝¹⁰ : CommRing R
      inst✝⁹ : AddCommGroup M
      inst✝⁸ : Module R M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R N
      K : Type u_4
      V : Type u_5
      W : Type u_6
      inst✝⁵ : Field K
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module K V
      inst✝² : AddCommGroup W
      inst✝¹ : Module K W
      inst✝ : FiniteDimensional K W
      B : LinearMap (RingHom.id K) V (LinearMap (RingHom.id K) W K)
      h : Function.Injective ⇑B
      h' : Function.Injective ⇑B.flip
      ⊢ Function.Surjective ⇑B
    -/
    have : FiniteDimensional K V := FiniteDimensional.of_injective B h
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝¹⁰ : CommRing R
      inst✝⁹ : AddCommGroup M
      inst✝⁸ : Module R M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R N
      K : Type u_4
      V : Type u_5
      W : Type u_6
      inst✝⁵ : Field K
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module K V
      inst✝² : AddCommGroup W
      inst✝¹ : Module K W
      inst✝ : FiniteDimensional K W
      B : LinearMap (RingHom.id K) V (LinearMap (RingHom.id K) W K)
      h : Function.Injective ⇑B
      h' : Function.Injective ⇑B.flip
      this : FiniteDimensional K V
      ⊢ Function.Surjective ⇑B
    -/
    rwa [← B.flip_injective_iff₁]⟩
    /-
      🎉 no goals
    -/
                            /-
                              R : Type u_1
                              M : Type u_2
                              N : Type u_3
                              inst✝¹⁰ : CommRing R
                              inst✝⁹ : AddCommGroup M
                              inst✝⁸ : Module R M
                              inst✝⁷ : AddCommGroup N
                              inst✝⁶ : Module R N
                              K : Type u_4
                              V : Type u_5
                              W : Type u_6
                              inst✝⁵ : Field K
                              inst✝⁴ : AddCommGroup V
                              inst✝³ : Module K V
                              inst✝² : AddCommGroup W
                              inst✝¹ : Module K W
                              inst✝ : FiniteDimensional K W
                              B : LinearMap (RingHom.id K) V (LinearMap (RingHom.id K) W K)
                              h : Function.Injective ⇑B
                              h' : Function.Injective ⇑B.flip
                              ⊢ Function.Surjective ⇑B.flip
                            -/
  bijectiveRight := ⟨h', by rwa [← B.flip.flip_injective_iff₁, LinearMap.flip_flip]⟩
                            /-
                              🎉 no goals
                            -/


instance instFunLike : FunLike (PerfectPairing R M N) M (N →ₗ[R] R) where
  coe f := f.toLin
                             /-
                               R : Type u_1
                               M : Type u_2
                               N : Type u_3
                               inst✝⁴ : CommRing R
                               inst✝³ : AddCommGroup M
                               inst✝² : Module R M
                               inst✝¹ : AddCommGroup N
                               inst✝ : Module R N
                               x y : PerfectPairing R M N
                               h : Eq ((fun f => ⇑f.toLin) x) ((fun f => ⇑f.toLin) y)
                               ⊢ Eq x y
                             -/
  coe_injective' x y h := by cases x; cases y; simpa using h
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
lemma toLin_apply (p : PerfectPairing R M N) {x : M} : p.toLin x = p x := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    x : M
    ⊢ Eq (p.toLin x) (p x)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a perfect pairing between `M` and `N`, we may interchange the roles of `M` and `N`. -/
protected def flip : PerfectPairing R N M where
  toLin := p.toLin.flip
  bijectiveLeft := p.bijectiveRight
  bijectiveRight := p.bijectiveLeft


@[simp]
lemma flip_apply_apply {x : M} {y : N} : p.flip y x = p x y :=
  rfl


@[simp]
lemma flip_flip : p.flip.flip = p :=
  rfl


/-- The linear equivalence from `M` to `Dual R N` induced by a perfect pairing. -/
def toDualLeft : M ≃ₗ[R] Dual R N :=
  LinearEquiv.ofBijective p.toLin p.bijectiveLeft


@[simp]
theorem toDualLeft_apply (a : M) : p.toDualLeft a = p a :=
  rfl


@[simp]
theorem apply_toDualLeft_symm_apply (f : Dual R N) (x : N) : p (p.toDualLeft.symm f) x = f x := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    f : Module.Dual R N
    x : N
    ⊢ Eq ((p (p.toDualLeft.symm f)) x) (f x)
  -/
  have h := LinearEquiv.apply_symm_apply p.toDualLeft f
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    f : Module.Dual R N
    x : N
    h : Eq (p.toDualLeft (p.toDualLeft.symm f)) f
    ⊢ Eq ((p (p.toDualLeft.symm f)) x) (f x)
  -/
  rw [toDualLeft_apply] at h
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    f : Module.Dual R N
    x : N
    h : Eq (p (p.toDualLeft.symm f)) f
    ⊢ Eq ((p (p.toDualLeft.symm f)) x) (f x)
  -/
  exact congrFun (congrArg DFunLike.coe h) x
  /-
    🎉 no goals
  -/


/-- The linear equivalence from `N` to `Dual R M` induced by a perfect pairing. -/
def toDualRight : N ≃ₗ[R] Dual R M :=
  toDualLeft p.flip


@[simp]
theorem toDualRight_apply (a : N) : p.toDualRight a = p.flip a :=
  rfl


@[simp]
theorem apply_apply_toDualRight_symm (x : M) (f : Dual R M) :
    (p x) (p.toDualRight.symm f) = f x := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    x : M
    f : Module.Dual R M
    ⊢ Eq ((p x) (p.toDualRight.symm f)) (f x)
  -/
  have h := LinearEquiv.apply_symm_apply p.toDualRight f
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    x : M
    f : Module.Dual R M
    h : Eq (p.toDualRight (p.toDualRight.symm f)) f
    ⊢ Eq ((p x) (p.toDualRight.symm f)) (f x)
  -/
  rw [toDualRight_apply] at h
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    x : M
    f : Module.Dual R M
    h : Eq (p.flip (p.toDualRight.symm f)) f
    ⊢ Eq ((p x) (p.toDualRight.symm f)) (f x)
  -/
  exact congrFun (congrArg DFunLike.coe h) x
  /-
    🎉 no goals
  -/


theorem toDualLeft_of_toDualRight_symm (x : M) (f : Dual R M) :
    (p.toDualLeft x) (p.toDualRight.symm f) = f x := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    x : M
    f : Module.Dual R M
    ⊢ Eq ((p.toDualLeft x) (p.toDualRight.symm f)) (f x)
  -/
  rw [@toDualLeft_apply]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    x : M
    f : Module.Dual R M
    ⊢ Eq ((p x) (p.toDualRight.symm f)) (f x)
  -/
  exact apply_apply_toDualRight_symm p x f
  /-
    🎉 no goals
  -/


theorem toDualRight_symm_toDualLeft (x : M) :
    p.toDualRight.symm.dualMap (p.toDualLeft x) = Dual.eval R M x := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    x : M
    ⊢ Eq (p.toDualRight.symm.dualMap (p.toDualLeft x)) ((Module.Dual.eval R M) x)
  -/
  ext f
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    x : M
    f : Module.Dual R M
    ⊢ Eq ((p.toDualRight.symm.dualMap (p.toDualLeft x)) f) (((Module.Dual.eval R M …
  -/
  simp only [LinearEquiv.dualMap_apply, Dual.eval_apply]
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    x : M
    f : Module.Dual R M
    ⊢ Eq ((p.toDualLeft x) (p.toDualRight.symm f)) (f x)
  -/
  exact toDualLeft_of_toDualRight_symm p x f
  /-
    🎉 no goals
  -/


theorem toDualRight_symm_comp_toDualLeft :
    p.toDualRight.symm.dualMap ∘ₗ (p.toDualLeft : M →ₗ[R] Dual R N) = Dual.eval R M := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    ⊢ Eq ((↑p.toDualRight.symm.dualMap).comp ↑p.toDualLeft) (Module.Dual.eval R M)
  -/
  ext1 x
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : PerfectPairing R M N
    x : M
    ⊢ Eq (((↑p.toDualRight.symm.dualMap).comp ↑p.toDualLeft) x) ((Module.Dual.eval …
  -/
  exact p.toDualRight_symm_toDualLeft x
  /-
    🎉 no goals
  -/


theorem bijective_toDualRight_symm_toDualLeft :
    Bijective (fun x => p.toDualRight.symm.dualMap (p.toDualLeft x)) :=
  Bijective.comp (LinearEquiv.bijective p.toDualRight.symm.dualMap)
    (LinearEquiv.bijective p.toDualLeft)


include p in
theorem reflexive_left : IsReflexive R M where
  bijective_dual_eval' := by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      p : PerfectPairing R M N
      ⊢ Function.Bijective ⇑(Module.Dual.eval R M)
    -/
    rw [← p.toDualRight_symm_comp_toDualLeft]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      p : PerfectPairing R M N
      ⊢ Function.Bijective ⇑((↑p.toDualRight.symm.dualMap).comp ↑p.toDualLeft)
    -/
    exact p.bijective_toDualRight_symm_toDualLeft
    /-
      🎉 no goals
    -/


include p in
theorem reflexive_right : IsReflexive R N :=
  p.flip.reflexive_left


instance : EquivLike (PerfectPairing R M N) M (Dual R N) where
  coe p := p.toDualLeft
  inv p := p.toDualLeft.symm
  left_inv p x := LinearEquiv.symm_apply_apply _ _
  right_inv p x := LinearEquiv.apply_symm_apply _ _
  coe_injective' p q h h' := by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      p✝ p q : PerfectPairing R M N
      h : Eq ((fun p => ⇑p.toDualLeft) p) ((fun p => ⇑p.toDualLeft) q)
      h' : Eq ((fun p => ⇑p.toDualLeft.symm) p) ((fun p => ⇑p.toDualLeft.symm) q)
      ⊢ Eq p q
    -/
    cases p
    /-
      case mk
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      p q : PerfectPairing R M N
      toLin✝ : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N R)
      bijectiveLeft✝ : Function.Bijective ⇑toLin✝
      bijectiveRight✝ : Function.Bijective ⇑toLin✝.flip
      h : Eq ((fun p => ⇑p.toDualLeft) { toLin := toLin✝, bijectiveLeft := bijective …
      h' : Eq ((fun p => ⇑p.toDualLeft.symm) { toLin := toLin✝, bijectiveLeft := bij …
      ⊢ Eq { toLin := toLin✝, bijectiveLeft := bijectiveLeft✝, bijectiveRight := bij …
    -/
    cases q
    /-
      case mk.mk
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      p : PerfectPairing R M N
      toLin✝¹ : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N R)
      bijectiveLeft✝¹ : Function.Bijective ⇑toLin✝¹
      bijectiveRight✝¹ : Function.Bijective ⇑toLin✝¹.flip
      toLin✝ : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N R)
      bijectiveLeft✝ : Function.Bijective ⇑toLin✝
      bijectiveRight✝ : Function.Bijective ⇑toLin✝.flip
      h : Eq ((fun p => ⇑p.toDualLeft) { toLin := toLin✝¹, bijectiveLeft := bijectiv …
      h' : Eq ((fun p => ⇑p.toDualLeft.symm) { toLin := toLin✝¹, bijectiveLeft := bi …
      ⊢ Eq { toLin := toLin✝¹, bijectiveLeft := bijectiveLeft✝¹, bijectiveRight := b …
    -/
    simp only [mk.injEq]
    /-
      case mk.mk
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      p : PerfectPairing R M N
      toLin✝¹ : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N R)
      bijectiveLeft✝¹ : Function.Bijective ⇑toLin✝¹
      bijectiveRight✝¹ : Function.Bijective ⇑toLin✝¹.flip
      toLin✝ : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N R)
      bijectiveLeft✝ : Function.Bijective ⇑toLin✝
      bijectiveRight✝ : Function.Bijective ⇑toLin✝.flip
      h : Eq ((fun p => ⇑p.toDualLeft) { toLin := toLin✝¹, bijectiveLeft := bijectiv …
      h' : Eq ((fun p => ⇑p.toDualLeft.symm) { toLin := toLin✝¹, bijectiveLeft := bi …
      ⊢ Eq toLin✝¹ toLin✝
    -/
    ext m n
    /-
      case mk.mk.h.h
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      p : PerfectPairing R M N
      toLin✝¹ : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N R)
      bijectiveLeft✝¹ : Function.Bijective ⇑toLin✝¹
      bijectiveRight✝¹ : Function.Bijective ⇑toLin✝¹.flip
      toLin✝ : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N R)
      bijectiveLeft✝ : Function.Bijective ⇑toLin✝
      bijectiveRight✝ : Function.Bijective ⇑toLin✝.flip
      h : Eq ((fun p => ⇑p.toDualLeft) { toLin := toLin✝¹, bijectiveLeft := bijectiv …
      h' : Eq ((fun p => ⇑p.toDualLeft.symm) { toLin := toLin✝¹, bijectiveLeft := bi …
      m : M
      n : N
      ⊢ Eq ((toLin✝¹ m) n) ((toLin✝ m) n)
    -/
    simp only [DFunLike.coe_fn_eq] at h
    /-
      case mk.mk.h.h
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      p : PerfectPairing R M N
      toLin✝¹ : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N R)
      bijectiveLeft✝¹ : Function.Bijective ⇑toLin✝¹
      bijectiveRight✝¹ : Function.Bijective ⇑toLin✝¹.flip
      toLin✝ : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N R)
      bijectiveLeft✝ : Function.Bijective ⇑toLin✝
      bijectiveRight✝ : Function.Bijective ⇑toLin✝.flip
      h' : Eq ((fun p => ⇑p.toDualLeft.symm) { toLin := toLin✝¹, bijectiveLeft := bi …
      m : M
      n : N
      h : Eq { toLin := toLin✝¹, bijectiveLeft := bijectiveLeft✝¹, bijectiveRight := …
      ⊢ Eq ((toLin✝¹ m) n) ((toLin✝ m) n)
    -/
    exact LinearMap.congr_fun (LinearEquiv.congr_fun h m) n
    /-
      🎉 no goals
    -/


instance : LinearEquivClass (PerfectPairing R M N) R M (Dual R N) where
  map_add p m₁ m₂ := p.toLin.map_add m₁ m₂
  map_smulₛₗ p t m := p.toLin.map_smul t m


include p in
theorem finrank_eq [Module.Finite R M] [Module.Free R M] :
    finrank R M = finrank R N :=
  ((Module.Free.chooseBasis R M).toDualEquiv.trans p.toDualRight.symm).finrank_eq


private lemma restrict_aux
    {M' N' : Type*} [AddCommGroup M'] [Module R M'] [AddCommGroup N'] [Module R N']
    (i : M' →ₗ[R] M) (j : N' →ₗ[R] N)
    (hM : IsCompl (LinearMap.range i) ((LinearMap.range j).dualAnnihilator.map p.toDualLeft.symm))
    (hN : IsCompl (LinearMap.range j) ((LinearMap.range i).dualAnnihilator.map p.toDualRight.symm))
    (hi : Injective i) (hj : Injective j) :
    Bijective (p.toLin.compl₁₂ i j) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    p : PerfectPairing R M N
    M' : Type u_4
    N' : Type u_5
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module R N'
    i : LinearMap (RingHom.id R) M' M
    j : LinearMap (RingHom.id R) N' N
    hM : IsCompl (LinearMap.range i) (Submodule.map p.toDualLeft.symm (LinearMap.r …
    hN : IsCompl (LinearMap.range j) (Submodule.map p.toDualRight.symm (LinearMap. …
    hi : Function.Injective ⇑i
    hj : Function.Injective ⇑j
    ⊢ Function.Bijective ⇑(p.toLin.compl₁₂ i j)
  -/
  refine ⟨LinearMap.ker_eq_bot.mp <| eq_bot_iff.mpr fun m hm ↦ ?_, fun f ↦ ?_⟩
  · replace hm : i m ∈ (LinearMap.range j).dualAnnihilator.map p.toDualLeft.symm := by
      simp only [Submodule.mem_map, Submodule.mem_dualAnnihilator]
      refine ⟨p.toDualLeft (i m), ?_, LinearEquiv.symm_apply_apply _ _⟩
      rintro - ⟨n, rfl⟩
      simpa using LinearMap.congr_fun hm n
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      p : PerfectPairing R M N
      M' : Type u_4
      N' : Type u_5
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      i : LinearMap (RingHom.id R) M' M
      j : LinearMap (RingHom.id R) N' N
      hM : IsCompl (LinearMap.range i) (Submodule.map p.toDualLeft.symm (LinearMap.r …
      hN : IsCompl (LinearMap.range j) (Submodule.map p.toDualRight.symm (LinearMap. …
      hi : Function.Injective ⇑i
      hj : Function.Injective ⇑j
      m : M'
      hm : Membership.mem (Submodule.map p.toDualLeft.symm (LinearMap.range j).dualA …
      ⊢ Membership.mem Bot.bot m
    -/
    suffices i m ∈ (⊥ : Submodule R M) by simpa [hi] using this
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      p : PerfectPairing R M N
      M' : Type u_4
      N' : Type u_5
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      i : LinearMap (RingHom.id R) M' M
      j : LinearMap (RingHom.id R) N' N
      hM : IsCompl (LinearMap.range i) (Submodule.map p.toDualLeft.symm (LinearMap.r …
      hN : IsCompl (LinearMap.range j) (Submodule.map p.toDualRight.symm (LinearMap. …
      hi : Function.Injective ⇑i
      hj : Function.Injective ⇑j
      m : M'
      hm : Membership.mem (Submodule.map p.toDualLeft.symm (LinearMap.range j).dualA …
      ⊢ Membership.mem Bot.bot (i m)
    -/
    simpa only [← hM.inf_eq_bot, Submodule.mem_inf] using ⟨LinearMap.mem_range_self i m, hm⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      p : PerfectPairing R M N
      M' : Type u_4
      N' : Type u_5
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      i : LinearMap (RingHom.id R) M' M
      j : LinearMap (RingHom.id R) N' N
      hM : IsCompl (LinearMap.range i) (Submodule.map p.toDualLeft.symm (LinearMap.r …
      hN : IsCompl (LinearMap.range j) (Submodule.map p.toDualRight.symm (LinearMap. …
      hi : Function.Injective ⇑i
      hj : Function.Injective ⇑j
      f : LinearMap (RingHom.id R) N' R
      ⊢ Exists fun a => Eq ((p.toLin.compl₁₂ i j) a) f
    -/
  · set F : Module.Dual R N := f ∘ₗ j.linearProjOfIsCompl _ hj hN with hF
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      p : PerfectPairing R M N
      M' : Type u_4
      N' : Type u_5
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      i : LinearMap (RingHom.id R) M' M
      j : LinearMap (RingHom.id R) N' N
      hM : IsCompl (LinearMap.range i) (Submodule.map p.toDualLeft.symm (LinearMap.r …
      hN : IsCompl (LinearMap.range j) (Submodule.map p.toDualRight.symm (LinearMap. …
      hi : Function.Injective ⇑i
      hj : Function.Injective ⇑j
      f : LinearMap (RingHom.id R) N' R
      F : Module.Dual R N := f.comp (LinearMap.linearProjOfIsCompl (Submodule.map p. …
      hF : Eq F (f.comp (LinearMap.linearProjOfIsCompl (Submodule.map p.toDualRight. …
      ⊢ Exists fun a => Eq ((p.toLin.compl₁₂ i j) a) f
    -/
    have hF (n : N') : F (j n) = f n := by simp [hF]
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      p : PerfectPairing R M N
      M' : Type u_4
      N' : Type u_5
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      i : LinearMap (RingHom.id R) M' M
      j : LinearMap (RingHom.id R) N' N
      hM : IsCompl (LinearMap.range i) (Submodule.map p.toDualLeft.symm (LinearMap.r …
      hN : IsCompl (LinearMap.range j) (Submodule.map p.toDualRight.symm (LinearMap. …
      hi : Function.Injective ⇑i
      hj : Function.Injective ⇑j
      f : LinearMap (RingHom.id R) N' R
      F : Module.Dual R N := f.comp (LinearMap.linearProjOfIsCompl (Submodule.map p. …
      hF✝ : Eq F (f.comp (LinearMap.linearProjOfIsCompl (Submodule.map p.toDualRight …
      hF : ∀ (n : N'), Eq (F (j n)) (f n)
      ⊢ Exists fun a => Eq ((p.toLin.compl₁₂ i j) a) f
    -/
    set m : M := p.toDualLeft.symm F with hm
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      p : PerfectPairing R M N
      M' : Type u_4
      N' : Type u_5
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      i : LinearMap (RingHom.id R) M' M
      j : LinearMap (RingHom.id R) N' N
      hM : IsCompl (LinearMap.range i) (Submodule.map p.toDualLeft.symm (LinearMap.r …
      hN : IsCompl (LinearMap.range j) (Submodule.map p.toDualRight.symm (LinearMap. …
      hi : Function.Injective ⇑i
      hj : Function.Injective ⇑j
      f : LinearMap (RingHom.id R) N' R
      F : Module.Dual R N := f.comp (LinearMap.linearProjOfIsCompl (Submodule.map p. …
      hF✝ : Eq F (f.comp (LinearMap.linearProjOfIsCompl (Submodule.map p.toDualRight …
      hF : ∀ (n : N'), Eq (F (j n)) (f n)
      m : M := p.toDualLeft.symm F
      hm : Eq m (p.toDualLeft.symm F)
      ⊢ Exists fun a => Eq ((p.toLin.compl₁₂ i j) a) f
    -/
    obtain ⟨-, ⟨m₀, rfl⟩, y, hy, hm'⟩ := Submodule.exists_add_eq_of_codisjoint hM.codisjoint m
    /-
      case refine_2.intro.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      p : PerfectPairing R M N
      M' : Type u_4
      N' : Type u_5
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      i : LinearMap (RingHom.id R) M' M
      j : LinearMap (RingHom.id R) N' N
      hM : IsCompl (LinearMap.range i) (Submodule.map p.toDualLeft.symm (LinearMap.r …
      hN : IsCompl (LinearMap.range j) (Submodule.map p.toDualRight.symm (LinearMap. …
      hi : Function.Injective ⇑i
      hj : Function.Injective ⇑j
      f : LinearMap (RingHom.id R) N' R
      F : Module.Dual R N := f.comp (LinearMap.linearProjOfIsCompl (Submodule.map p. …
      hF✝ : Eq F (f.comp (LinearMap.linearProjOfIsCompl (Submodule.map p.toDualRight …
      hF : ∀ (n : N'), Eq (F (j n)) (f n)
      m : M := p.toDualLeft.symm F
      hm : Eq m (p.toDualLeft.symm F)
      m₀ : M'
      y : M
      hy : Membership.mem (Submodule.map p.toDualLeft.symm (LinearMap.range j).dualA …
      hm' : Eq (HAdd.hAdd (i m₀) y) m
      ⊢ Exists fun a => Eq ((p.toLin.compl₁₂ i j) a) f
    -/
    refine ⟨m₀, LinearMap.ext fun n ↦ ?_⟩
    replace hy : (p y) (j n) = 0 := by
      simp only [Submodule.mem_map, Submodule.mem_dualAnnihilator] at hy
      obtain ⟨g, hg, rfl⟩ := hy
      simpa only [apply_toDualLeft_symm_apply] using hg _ (LinearMap.mem_range_self j n)
    rw [hm, ← LinearEquiv.symm_apply_eq, map_add, LinearEquiv.symm_symm,
      toDualLeft_apply] at hm'
    /-
      case refine_2.intro.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      p : PerfectPairing R M N
      M' : Type u_4
      N' : Type u_5
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      i : LinearMap (RingHom.id R) M' M
      j : LinearMap (RingHom.id R) N' N
      hM : IsCompl (LinearMap.range i) (Submodule.map p.toDualLeft.symm (LinearMap.r …
      hN : IsCompl (LinearMap.range j) (Submodule.map p.toDualRight.symm (LinearMap. …
      hi : Function.Injective ⇑i
      hj : Function.Injective ⇑j
      f : LinearMap (RingHom.id R) N' R
      F : Module.Dual R N := f.comp (LinearMap.linearProjOfIsCompl (Submodule.map p. …
      hF✝ : Eq F (f.comp (LinearMap.linearProjOfIsCompl (Submodule.map p.toDualRight …
      hF : ∀ (n : N'), Eq (F (j n)) (f n)
      m : M := p.toDualLeft.symm F
      hm : Eq m (p.toDualLeft.symm F)
      m₀ : M'
      y : M
      hm' : Eq (HAdd.hAdd (p (i m₀)) (p.toDualLeft y)) F
      n : N'
      hy : Eq ((p y) (j n)) 0
      ⊢ Eq (((p.toLin.compl₁₂ i j) m₀) n) (f n)
    -/
    simpa [← hF, ← LinearMap.congr_fun hm' (j n)]
    /-
      🎉 no goals
    -/


/-- The restriction of a perfect pairing to submodules (expressed as injections to provide
definitional control). -/
@[simps]
def restrict {M' N' : Type*} [AddCommGroup M'] [Module R M'] [AddCommGroup N'] [Module R N']
    (i : M' →ₗ[R] M) (j : N' →ₗ[R] N)
    (hM : IsCompl (LinearMap.range i) ((LinearMap.range j).dualAnnihilator.map p.toDualLeft.symm))
    (hN : IsCompl (LinearMap.range j) ((LinearMap.range i).dualAnnihilator.map p.toDualRight.symm))
    (hi : Injective i) (hj : Injective j) :
    PerfectPairing R M' N' where
  toLin := p.toLin.compl₁₂ i j
  bijectiveLeft := p.restrict_aux i j hM hN hi hj
  bijectiveRight := p.flip.restrict_aux j i hN hM hj hi


/-- If a perfect pairing over a field `L` takes values in a subfield `K` along two `K`-subspaces
whose `L` span is full, then these subspaces induce a `K`-structure in the sense of
[*Algebra I*, Bourbaki : Chapter II, §8.1 Definition 1][bourbaki1989]. -/
lemma exists_basis_basis_of_span_eq_top_of_mem_algebraMap
    {K L : Type*} [Field K] [Field L] [Algebra K L]
    [Module L M] [Module L N] [Module K M] [Module K N] [IsScalarTower K L M]
    (p : PerfectPairing L M N)
    (M' : Submodule K M) (N' : Submodule K N)
    (hM : span L (M' : Set M) = ⊤)
    (hN : span L (N' : Set N) = ⊤)
    (hp : ∀ᵉ (x ∈ M') (y ∈ N'), p x y ∈ (algebraMap K L).range) :
    ∃ (n : ℕ) (b : Basis (Fin n) L M) (b' : Basis (Fin n) K M'), ∀ i, b i = b' i := by
  classical
  have : IsReflexive L M := p.reflexive_left
  have : IsReflexive L N := p.reflexive_right
  obtain ⟨v, hv₁, hv₂, hv₃⟩ := exists_linearIndependent L (M' : Set M)
  rw [hM] at hv₂
  let b : Basis _ L M := Basis.mk hv₃ <| by rw [← hv₂, Subtype.range_coe_subtype, Set.setOf_mem_eq]
  have : Fintype v := Set.Finite.fintype <| Module.Finite.finite_basis b
  set v' : v → M' := fun i ↦ ⟨i, hv₁ (Subtype.coe_prop i)⟩
  have hv' : LinearIndependent K v' := by
    replace hv₃ := hv₃.restrict_scalars (R := K) <| by
      simp_rw [← Algebra.algebraMap_eq_smul_one]
      exact NoZeroSMulDivisors.algebraMap_injective K L
    rw [show ((↑) : v → M) = M'.subtype ∘ v' from rfl] at hv₃
    exact hv₃.of_comp
  suffices span K (Set.range v') = ⊤ by
    let e := (Module.Finite.finite_basis b).equivFin
    let b' : Basis _ K M' := Basis.mk hv' (by rw [this])
    exact ⟨_, b.reindex e, b'.reindex e, fun i ↦ by simp [b, b', v']⟩
  suffices span K v = M' by
    apply Submodule.map_injective_of_injective M'.injective_subtype
    rw [Submodule.map_span, ← Set.image_univ, Set.image_image]
    simpa [v']
  refine le_antisymm (Submodule.span_le.mpr hv₁) fun m hm ↦ ?_
  obtain ⟨w, hw₁, hw₂, hw₃⟩ := exists_linearIndependent L (N' : Set N)
  rw [hN] at hw₂
  let bN : Basis _ L N := Basis.mk hw₃ <| by rw [← hw₂, Subtype.range_coe_subtype, Set.setOf_mem_eq]
  have : Fintype w := Set.Finite.fintype <| Module.Finite.finite_basis bN
  have e : v ≃ w := Fintype.equivOfCardEq <| by rw [← Module.finrank_eq_card_basis b,
    ← Module.finrank_eq_card_basis bN, p.finrank_eq]
  let bM := bN.dualBasis.map p.toDualLeft.symm
  have hbM (j : w) (x : M) (hx : x ∈ M') : bM.repr x j = p x (j : N) := by simp [bM, bN]
  have hj (j : w) : bM.repr m j ∈ (algebraMap K L).range := (hbM _ _ hm) ▸ hp m hm j (hw₁ j.2)
  replace hp (i : w) (j : v) :
      (bN.dualBasis.map p.toDualLeft.symm).toMatrix b i j ∈ (algebraMap K L).fieldRange := by
    simp only [Basis.toMatrix, Basis.map_repr, LinearEquiv.symm_symm, LinearEquiv.trans_apply,
      toDualLeft_apply, Basis.dualBasis_repr]
    exact hp (b j) (by simpa [b] using hv₁ j.2) (bN i) (by simpa [bN] using hw₁ i.2)
  have hA (i j) : b.toMatrix bM i j ∈ (algebraMap K L).range :=
    Matrix.mem_subfield_of_mul_eq_one_of_mem_subfield_left e _ (by simp [bM]) hp i j
  have h_span : span K v = span K (Set.range b) := by simp [b]
  rw [h_span, Basis.mem_span_iff_repr_mem, ← Basis.toMatrix_mulVec_repr bM b m]
  exact fun i ↦ Subring.sum_mem _ fun j _ ↦ Subring.mul_mem _ (hA i j) (hj j)


/-- An auxiliary definition used to construct `PerfectPairing.restrictScalars`. -/
private def restrictScalarsAux
    (hp : ∀ m n, p (i m) (j n) ∈ (algebraMap S R).range) :
    M' →ₗ[S] N' →ₗ[S] S :=
 LinearMap.restrictScalarsRange i j (Algebra.linearMap S R)
    (NoZeroSMulDivisors.algebraMap_injective S R) p.toLin hp


private lemma restrictScalarsAux_injective
    (hi : Injective i)
    (hN : span R (LinearMap.range j : Set N) = ⊤)
    (hp : ∀ m n, p (i m) (j n) ∈ (algebraMap S R).range) :
    Injective (p.restrictScalarsAux i j hp) := by
  let f := LinearMap.restrictScalarsRange i j (Algebra.linearMap S R)
      (NoZeroSMulDivisors.algebraMap_injective S R) p.toLin hp
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    hi : Function.Injective ⇑i
    hN : Eq (Submodule.span R ↑(LinearMap.range j)) Top.top
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    f : LinearMap (RingHom.id S) M' (LinearMap (RingHom.id S) N' S) := i.restrictS …
    ⊢ Function.Injective ⇑(PerfectPairing.restrictScalarsAux p i j hp)
  -/
  rw [← LinearMap.ker_eq_bot]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    hi : Function.Injective ⇑i
    hN : Eq (Submodule.span R ↑(LinearMap.range j)) Top.top
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    f : LinearMap (RingHom.id S) M' (LinearMap (RingHom.id S) N' S) := i.restrictS …
    ⊢ Eq (LinearMap.ker (PerfectPairing.restrictScalarsAux p i j hp)) Bot.bot
  -/
  refine (Submodule.eq_bot_iff _).mpr fun x (hx : f x = 0) ↦ ?_
  replace hx (n : N) : p (i x) n = 0 := by
    have hn : n ∈ span R (LinearMap.range j : Set N) := hN ▸ Submodule.mem_top
    induction' hn using Submodule.span_induction with z hz
    · obtain ⟨n', rfl⟩ := hz
      simpa [f] using LinearMap.congr_fun hx n'
    · simp
    · rw [← p.toLin_apply, map_add]; aesop
    · rw [← p.toLin_apply, map_smul]; aesop
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    hi : Function.Injective ⇑i
    hN : Eq (Submodule.span R ↑(LinearMap.range j)) Top.top
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    f : LinearMap (RingHom.id S) M' (LinearMap (RingHom.id S) N' S) := i.restrictS …
    x : M'
    hx : ∀ (n : N), Eq ((p (i x)) n) 0
    ⊢ Eq x 0
  -/
  rw [← i.map_eq_zero_iff hi, ← p.toLin.map_eq_zero_iff p.bijectiveLeft.injective]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    hi : Function.Injective ⇑i
    hN : Eq (Submodule.span R ↑(LinearMap.range j)) Top.top
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    f : LinearMap (RingHom.id S) M' (LinearMap (RingHom.id S) N' S) := i.restrictS …
    x : M'
    hx : ∀ (n : N), Eq ((p (i x)) n) 0
    ⊢ Eq (p.toLin (i x)) 0
  -/
  ext n
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    hi : Function.Injective ⇑i
    hN : Eq (Submodule.span R ↑(LinearMap.range j)) Top.top
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    f : LinearMap (RingHom.id S) M' (LinearMap (RingHom.id S) N' S) := i.restrictS …
    x : M'
    hx : ∀ (n : N), Eq ((p (i x)) n) 0
    n : N
    ⊢ Eq ((p.toLin (i x)) n) (0 n)
  -/
  simpa using hx n
  /-
    🎉 no goals
  -/


private lemma restrictScalarsAux_surjective
    (h : ∀ g : Module.Dual S N', ∃ m,
      (p.toDualLeft (i m)).restrictScalars S ∘ₗ j = Algebra.linearMap S R ∘ₗ g)
    (hp : ∀ m n, p (i m) (j n) ∈ (algebraMap S R).range) :
    Surjective (p.restrictScalarsAux i j hp) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    h : ∀ (g : Module.Dual S N'), Exists fun m => Eq ((↑S (p.toDualLeft (i m))).co …
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    ⊢ Function.Surjective ⇑(PerfectPairing.restrictScalarsAux p i j hp)
  -/
  rw [← LinearMap.range_eq_top]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    h : ∀ (g : Module.Dual S N'), Exists fun m => Eq ((↑S (p.toDualLeft (i m))).co …
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    ⊢ Eq (LinearMap.range (PerfectPairing.restrictScalarsAux p i j hp)) Top.top
  -/
  refine Submodule.eq_top_iff'.mpr fun g : Module.Dual S N' ↦ ?_
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    h : ∀ (g : Module.Dual S N'), Exists fun m => Eq ((↑S (p.toDualLeft (i m))).co …
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    g : Module.Dual S N'
    ⊢ Membership.mem (LinearMap.range (PerfectPairing.restrictScalarsAux p i j hp) …
  -/
  obtain ⟨m, hm⟩ := h g
  /-
    case intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    h : ∀ (g : Module.Dual S N'), Exists fun m => Eq ((↑S (p.toDualLeft (i m))).co …
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    g : Module.Dual S N'
    m : M'
    hm : Eq ((↑S (p.toDualLeft (i m))).comp j) ((Algebra.linearMap S R).comp g)
    ⊢ Membership.mem (LinearMap.range (PerfectPairing.restrictScalarsAux p i j hp) …
  -/
  refine ⟨m, ?_⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    h : ∀ (g : Module.Dual S N'), Exists fun m => Eq ((↑S (p.toDualLeft (i m))).co …
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    g : Module.Dual S N'
    m : M'
    hm : Eq ((↑S (p.toDualLeft (i m))).comp j) ((Algebra.linearMap S R).comp g)
    ⊢ Eq ((PerfectPairing.restrictScalarsAux p i j hp) m) g
  -/
  ext n
  /-
    case intro.h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    h : ∀ (g : Module.Dual S N'), Exists fun m => Eq ((↑S (p.toDualLeft (i m))).co …
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    g : Module.Dual S N'
    m : M'
    hm : Eq ((↑S (p.toDualLeft (i m))).comp j) ((Algebra.linearMap S R).comp g)
    n : N'
    ⊢ Eq (((PerfectPairing.restrictScalarsAux p i j hp) m) n) (g n)
  -/
  apply NoZeroSMulDivisors.algebraMap_injective S R
  /-
    case intro.h.a
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    h : ∀ (g : Module.Dual S N'), Exists fun m => Eq ((↑S (p.toDualLeft (i m))).co …
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    g : Module.Dual S N'
    m : M'
    hm : Eq ((↑S (p.toDualLeft (i m))).comp j) ((Algebra.linearMap S R).comp g)
    n : N'
    ⊢ Eq ((algebraMap S R) (((PerfectPairing.restrictScalarsAux p i j hp) m) n)) ( …
  -/
  change Algebra.linearMap S R _ = _
  /-
    case intro.h.a
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    p : PerfectPairing R M N
    S : Type u_4
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Algebra S R
    inst✝⁹ : Module S M
    inst✝⁸ : Module S N
    inst✝⁷ : IsScalarTower S R M
    inst✝⁶ : IsScalarTower S R N
    inst✝⁵ : NoZeroSMulDivisors S R
    inst✝⁴ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝³ : AddCommGroup M'
    inst✝² : Module S M'
    inst✝¹ : AddCommGroup N'
    inst✝ : Module S N'
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    h : ∀ (g : Module.Dual S N'), Exists fun m => Eq ((↑S (p.toDualLeft (i m))).co …
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap S R).range ((p (i m)) (j  …
    g : Module.Dual S N'
    m : M'
    hm : Eq ((↑S (p.toDualLeft (i m))).comp j) ((Algebra.linearMap S R).comp g)
    n : N'
    ⊢ Eq ((Algebra.linearMap S R) (((PerfectPairing.restrictScalarsAux p i j hp) m …
  -/
  simpa [restrictScalarsAux] using LinearMap.congr_fun hm n
  /-
    🎉 no goals
  -/


/-- Restriction of scalars for a perfect pairing taking values in a subring. -/
def restrictScalars
    (hi : Injective i) (hj : Injective j)
    (hM : span R (LinearMap.range i : Set M) = ⊤)
    (hN : span R (LinearMap.range j : Set N) = ⊤)
    (h₁ : ∀ g : Module.Dual S N', ∃ m,
      (p.toDualLeft (i m)).restrictScalars S ∘ₗ j = Algebra.linearMap S R ∘ₗ g)
    (h₂ : ∀ g : Module.Dual S M', ∃ n,
      (p.toDualRight (j n)).restrictScalars S ∘ₗ i = Algebra.linearMap S R ∘ₗ g)
    (hp : ∀ m n, p (i m) (j n) ∈ (algebraMap S R).range) :
    PerfectPairing S M' N' :=
  { toLin := p.restrictScalarsAux i j hp
    bijectiveLeft := ⟨p.restrictScalarsAux_injective i j hi hN hp,
      p.restrictScalarsAux_surjective i j h₁ hp⟩
    bijectiveRight := ⟨p.flip.restrictScalarsAux_injective j i hj hM (fun m n ↦ hp n m),
      p.flip.restrictScalarsAux_surjective j i h₂ (fun m n ↦ hp n m)⟩}


/-- Restriction of scalars for a perfect pairing taking values in a subfield. -/
def restrictScalarsField {K L : Type*} [Field K] [Field L] [Algebra K L]
    [Module L M] [Module L N] [Module K M] [Module K N] [IsScalarTower K L M] [IsScalarTower K L N]
    [Module K M'] [Module K N']
    (i : M' →ₗ[K] M) (j : N' →ₗ[K] N)
    (hi : Injective i) (hj : Injective j)
    (hM : span L (LinearMap.range i : Set M) = ⊤)
    (hN : span L (LinearMap.range j : Set N) = ⊤)
    (p : PerfectPairing L M N)
    (hp : ∀ m n, p (i m) (j n) ∈ (algebraMap K L).range) :
    PerfectPairing K M' N' := by
  suffices FiniteDimensional K M' from mkOfInjective _ (p.restrictScalarsAux_injective i j hi hN hp)
    (p.flip.restrictScalarsAux_injective j i hj hM (fun m n ↦ hp n m))
  obtain ⟨n, -, b', -⟩ := p.exists_basis_basis_of_span_eq_top_of_mem_algebraMap _ _ hM hN <| by
    rintro - ⟨m, rfl⟩ - ⟨n, rfl⟩
    exact hp m n
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝²⁷ : CommRing R
    inst✝²⁶ : AddCommGroup M
    inst✝²⁵ : Module R M
    inst✝²⁴ : AddCommGroup N
    inst✝²³ : Module R N
    p✝ : PerfectPairing R M N
    S : Type u_4
    inst✝²² : CommRing S
    inst✝²¹ : Algebra S R
    inst✝²⁰ : Module S M
    inst✝¹⁹ : Module S N
    inst✝¹⁸ : IsScalarTower S R M
    inst✝¹⁷ : IsScalarTower S R N
    inst✝¹⁶ : NoZeroSMulDivisors S R
    inst✝¹⁵ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝¹⁴ : AddCommGroup M'
    inst✝¹³ : Module S M'
    inst✝¹² : AddCommGroup N'
    inst✝¹¹ : Module S N'
    i✝ : LinearMap (RingHom.id S) M' M
    j✝ : LinearMap (RingHom.id S) N' N
    K : Type u_7
    L : Type u_8
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra K L
    inst✝⁷ : Module L M
    inst✝⁶ : Module L N
    inst✝⁵ : Module K M
    inst✝⁴ : Module K N
    inst✝³ : IsScalarTower K L M
    inst✝² : IsScalarTower K L N
    inst✝¹ : Module K M'
    inst✝ : Module K N'
    i : LinearMap (RingHom.id K) M' M
    j : LinearMap (RingHom.id K) N' N
    hi : Function.Injective ⇑i
    hj : Function.Injective ⇑j
    hM : Eq (Submodule.span L ↑(LinearMap.range i)) Top.top
    hN : Eq (Submodule.span L ↑(LinearMap.range j)) Top.top
    p : PerfectPairing L M N
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap K L).range ((p (i m)) (j  …
    n : Nat
    b' : Basis (Fin n) K (Subtype fun x => Membership.mem (LinearMap.range i) x)
    ⊢ FiniteDimensional K M'
  -/
  have : FiniteDimensional K (LinearMap.range i) := FiniteDimensional.of_fintype_basis b'
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝²⁷ : CommRing R
    inst✝²⁶ : AddCommGroup M
    inst✝²⁵ : Module R M
    inst✝²⁴ : AddCommGroup N
    inst✝²³ : Module R N
    p✝ : PerfectPairing R M N
    S : Type u_4
    inst✝²² : CommRing S
    inst✝²¹ : Algebra S R
    inst✝²⁰ : Module S M
    inst✝¹⁹ : Module S N
    inst✝¹⁸ : IsScalarTower S R M
    inst✝¹⁷ : IsScalarTower S R N
    inst✝¹⁶ : NoZeroSMulDivisors S R
    inst✝¹⁵ : Nontrivial R
    M' : Type u_5
    N' : Type u_6
    inst✝¹⁴ : AddCommGroup M'
    inst✝¹³ : Module S M'
    inst✝¹² : AddCommGroup N'
    inst✝¹¹ : Module S N'
    i✝ : LinearMap (RingHom.id S) M' M
    j✝ : LinearMap (RingHom.id S) N' N
    K : Type u_7
    L : Type u_8
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra K L
    inst✝⁷ : Module L M
    inst✝⁶ : Module L N
    inst✝⁵ : Module K M
    inst✝⁴ : Module K N
    inst✝³ : IsScalarTower K L M
    inst✝² : IsScalarTower K L N
    inst✝¹ : Module K M'
    inst✝ : Module K N'
    i : LinearMap (RingHom.id K) M' M
    j : LinearMap (RingHom.id K) N' N
    hi : Function.Injective ⇑i
    hj : Function.Injective ⇑j
    hM : Eq (Submodule.span L ↑(LinearMap.range i)) Top.top
    hN : Eq (Submodule.span L ↑(LinearMap.range j)) Top.top
    p : PerfectPairing L M N
    hp : ∀ (m : M') (n : N'), Membership.mem (algebraMap K L).range ((p (i m)) (j  …
    n : Nat
    b' : Basis (Fin n) K (Subtype fun x => Membership.mem (LinearMap.range i) x)
    this : FiniteDimensional K (Subtype fun x => Membership.mem (LinearMap.range i …
    ⊢ FiniteDimensional K M'
  -/
  exact Finite.equiv (LinearEquiv.ofInjective i hi).symm
  /-
    🎉 no goals
  -/


/-- A reflexive module has a perfect pairing with its dual. -/
@[simps]
def IsReflexive.toPerfectPairingDual : PerfectPairing R (Dual R M) M where
  toLin := LinearMap.id
  bijectiveLeft := bijective_id
  bijectiveRight := bijective_dual_eval R M


@[simp]
lemma IsReflexive.toPerfectPairingDual_apply {f : Dual R M} {x : M} :
    IsReflexive.toPerfectPairingDual (R := R) f x = f x :=
  rfl


/-- For a reflexive module `M`, an equivalence `N ≃ₗ[R] Dual R M` naturally yields an equivalence
`M ≃ₗ[R] Dual R N`. Such equivalences are known as perfect pairings. -/
def flip : M ≃ₗ[R] Dual R N :=
  (evalEquiv R M).trans e.dualMap


@[simp] lemma coe_toLinearMap_flip : e.flip = (↑e : N →ₗ[R] Dual R M).flip := rfl


@[simp] lemma flip_apply (m : M) (n : N) : e.flip m n = e n m := rfl


lemma symm_flip : e.flip.symm = e.symm.dualMap.trans (evalEquiv R M).symm := rfl


lemma trans_dualMap_symm_flip : e.trans e.flip.symm.dualMap = Dual.eval R N := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    ⊢ Eq (↑(e.trans e.flip.symm.dualMap)) (Module.Dual.eval R N)
  -/
  ext; simp [symm_flip]
       /-
         🎉 no goals
       -/


include e in
/-- If `N` is in perfect pairing with `M`, then it is reflexive. -/
lemma isReflexive_of_equiv_dual_of_isReflexive : IsReflexive R N := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    ⊢ Module.IsReflexive R N
  -/
  constructor
  /-
    case bijective_dual_eval'
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    ⊢ Function.Bijective ⇑(Module.Dual.eval R N)
  -/
  rw [← trans_dualMap_symm_flip e]
  /-
    case bijective_dual_eval'
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    ⊢ Function.Bijective ⇑↑(e.trans e.flip.symm.dualMap)
  -/
  exact LinearEquiv.bijective _
  /-
    🎉 no goals
  -/


@[simp] lemma flip_flip (h : IsReflexive R N := isReflexive_of_equiv_dual_of_isReflexive e) :
    e.flip.flip = e := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    h : optParam (Module.IsReflexive R N) ⋯
    ⊢ Eq e.flip.flip e
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


/-- If `M` is reflexive then a linear equivalence `N ≃ Dual R M` is a perfect pairing. -/
@[simps]
def toPerfectPairing : PerfectPairing R N M where
  toLin := e
  bijectiveLeft := e.bijective
  bijectiveRight := e.flip.bijective


/-- A perfect pairing induces a perfect pairing between dual spaces. -/
def PerfectPairing.dual (p : PerfectPairing R M N) :
    PerfectPairing R (Dual R M) (Dual R N) :=
  let _i := p.reflexive_right
  (p.toDualRight.symm.trans (evalEquiv R N)).toPerfectPairing


@[simp]
lemma dualCoannihilator_map_linearEquiv_flip (p : Submodule R M) :
    (p.map e.flip).dualCoannihilator = p.dualAnnihilator.map e.symm := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    p : Submodule R M
    ⊢ Eq (Submodule.map e.flip p).dualCoannihilator (Submodule.map e.symm p.dualAn …
  -/
  ext; simp [LinearEquiv.symm_apply_eq, Submodule.mem_dualCoannihilator]
       /-
         🎉 no goals
       -/


@[simp]
lemma map_dualAnnihilator_linearEquiv_flip_symm (p : Submodule R N) :
    p.dualAnnihilator.map e.flip.symm = (p.map e).dualCoannihilator := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    p : Submodule R N
    ⊢ Eq (Submodule.map e.flip.symm p.dualAnnihilator) (Submodule.map e p).dualCoa …
  -/
  have : IsReflexive R N := e.isReflexive_of_equiv_dual_of_isReflexive
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    p : Submodule R N
    this : Module.IsReflexive R N
    ⊢ Eq (Submodule.map e.flip.symm p.dualAnnihilator) (Submodule.map e p).dualCoa …
  -/
  rw [← dualCoannihilator_map_linearEquiv_flip, flip_flip]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_dualCoannihilator_linearEquiv_flip (p : Submodule R (Dual R M)) :
    p.dualCoannihilator.map e.flip = (p.map e.symm).dualAnnihilator := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    p : Submodule R (Module.Dual R M)
    ⊢ Eq (Submodule.map e.flip p.dualCoannihilator) (Submodule.map e.symm p).dualA …
  -/
  have : IsReflexive R N := e.isReflexive_of_equiv_dual_of_isReflexive
  suffices (p.map e.symm).dualAnnihilator.map e.flip.symm =
      (p.dualCoannihilator.map e.flip).map e.flip.symm by
    exact (Submodule.map_injective_of_injective e.flip.symm.injective this).symm
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    p : Submodule R (Module.Dual R M)
    this : Module.IsReflexive R N
    ⊢ Eq (Submodule.map e.flip.symm (Submodule.map e.symm p).dualAnnihilator) (Sub …
  -/
  erw [← dualCoannihilator_map_linearEquiv_flip, flip_flip, ← map_comp, ← map_comp]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    p : Submodule R (Module.Dual R M)
    this : Module.IsReflexive R N
    ⊢ Eq (Submodule.map ((↑e).comp ↑e.symm) p).dualCoannihilator (Submodule.map (( …
  -/
  simp [-coe_toLinearMap_flip]
  /-
    🎉 no goals
  -/


@[simp]
lemma dualAnnihilator_map_linearEquiv_flip_symm (p : Submodule R (Dual R N)) :
    (p.map e.flip.symm).dualAnnihilator = p.dualCoannihilator.map e := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    p : Submodule R (Module.Dual R N)
    ⊢ Eq (Submodule.map e.flip.symm p).dualAnnihilator (Submodule.map e p.dualCoan …
  -/
  have : IsReflexive R N := e.isReflexive_of_equiv_dual_of_isReflexive
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.IsReflexive R M
    e : LinearEquiv (RingHom.id R) N (Module.Dual R M)
    p : Submodule R (Module.Dual R N)
    this : Module.IsReflexive R N
    ⊢ Eq (Submodule.map e.flip.symm p).dualAnnihilator (Submodule.map e p.dualCoan …
  -/
  rw [← map_dualCoannihilator_linearEquiv_flip, flip_flip]
  /-
    🎉 no goals
  -/


