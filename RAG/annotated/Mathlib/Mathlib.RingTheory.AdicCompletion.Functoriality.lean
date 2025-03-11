/-- `R`-linear version of `reduceModIdeal`. -/
private def reduceModIdealAux (f : M →ₗ[R] N) :
    M ⧸ (I • ⊤ : Submodule R M) →ₗ[R] N ⧸ (I • ⊤ : Submodule R N) :=
  Submodule.mapQ (I • ⊤ : Submodule R M) (I • ⊤ : Submodule R N) f
    (fun x hx ↦ by
      /-
        R : Type u_1
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u_2
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u_3
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u_4
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        T : Type u_5
        inst✝¹ : AddCommGroup T
        inst✝ : Module (AdicCompletion I R) T
        f : LinearMap (RingHom.id R) M N
        x : M
        hx : Membership.mem (HSMul.hSMul I Top.top) x
        ⊢ Membership.mem (Submodule.comap f (HSMul.hSMul I Top.top)) x
      -/
      refine Submodule.smul_induction_on hx (fun r hr x _ ↦ ?_) (fun x y hx hy ↦ ?_)
        /-
          case refine_1
          R : Type u_1
          inst✝⁸ : CommRing R
          I : Ideal R
          M : Type u_2
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          N : Type u_3
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          P : Type u_4
          inst✝³ : AddCommGroup P
          inst✝² : Module R P
          T : Type u_5
          inst✝¹ : AddCommGroup T
          inst✝ : Module (AdicCompletion I R) T
          f : LinearMap (RingHom.id R) M N
          x✝¹ : M
          hx : Membership.mem (HSMul.hSMul I Top.top) x✝¹
          r : R
          hr : Membership.mem I r
          x : M
          x✝ : Membership.mem Top.top x
          ⊢ Membership.mem (Submodule.comap f (HSMul.hSMul I Top.top)) (HSMul.hSMul r x)
        -/
      · simp [Submodule.smul_mem_smul hr Submodule.mem_top]
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          R : Type u_1
          inst✝⁸ : CommRing R
          I : Ideal R
          M : Type u_2
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          N : Type u_3
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          P : Type u_4
          inst✝³ : AddCommGroup P
          inst✝² : Module R P
          T : Type u_5
          inst✝¹ : AddCommGroup T
          inst✝ : Module (AdicCompletion I R) T
          f : LinearMap (RingHom.id R) M N
          x✝ : M
          hx✝ : Membership.mem (HSMul.hSMul I Top.top) x✝
          x y : M
          hx : Membership.mem (Submodule.comap f (HSMul.hSMul I Top.top)) x
          hy : Membership.mem (Submodule.comap f (HSMul.hSMul I Top.top)) y
          ⊢ Membership.mem (Submodule.comap f (HSMul.hSMul I Top.top)) (HAdd.hAdd x y)
        -/
      · simp [Submodule.add_mem _ hx hy])
        /-
          🎉 no goals
        -/


@[local simp]
private theorem reduceModIdealAux_apply (f : M →ₗ[R] N) (x : M) :
    (f.reduceModIdealAux I) (Submodule.Quotient.mk (p := (I • ⊤ : Submodule R M)) x) =
      Submodule.Quotient.mk (p := (I • ⊤ : Submodule R N)) (f x) :=
  rfl


/-- The induced linear map on the quotients mod `I • ⊤`. -/
def reduceModIdeal (f : M →ₗ[R] N) :
    M ⧸ (I • ⊤ : Submodule R M) →ₗ[R ⧸ I] N ⧸ (I • ⊤ : Submodule R N) where
  toFun := f.reduceModIdealAux I
                 /-
                   R : Type u_1
                   inst✝⁸ : CommRing R
                   I : Ideal R
                   M : Type u_2
                   inst✝⁷ : AddCommGroup M
                   inst✝⁶ : Module R M
                   N : Type u_3
                   inst✝⁵ : AddCommGroup N
                   inst✝⁴ : Module R N
                   P : Type u_4
                   inst✝³ : AddCommGroup P
                   inst✝² : Module R P
                   T : Type u_5
                   inst✝¹ : AddCommGroup T
                   inst✝ : Module (AdicCompletion I R) T
                   f : LinearMap (RingHom.id R) M N
                   ⊢ ∀ (x y : HasQuotient.Quotient M (HSMul.hSMul I Top.top)), Eq ((LinearMap.red …
                 -/
  map_add' := by simp
                 /-
                   🎉 no goals
                 -/
  map_smul' r x := by
    /-
      R : Type u_1
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u_3
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u_4
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      T : Type u_5
      inst✝¹ : AddCommGroup T
      inst✝ : Module (AdicCompletion I R) T
      f : LinearMap (RingHom.id R) M N
      r : HasQuotient.Quotient R I
      x : HasQuotient.Quotient M (HSMul.hSMul I Top.top)
      ⊢ Eq ({ toFun := ⇑(LinearMap.reduceModIdealAux I f), map_add' := ⋯ }.toFun (HS …
    -/
    refine Quotient.inductionOn' r (fun r ↦ ?_)
    /-
      R : Type u_1
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u_3
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u_4
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      T : Type u_5
      inst✝¹ : AddCommGroup T
      inst✝ : Module (AdicCompletion I R) T
      f : LinearMap (RingHom.id R) M N
      r✝ : HasQuotient.Quotient R I
      x : HasQuotient.Quotient M (HSMul.hSMul I Top.top)
      r : R
      ⊢ Eq ({ toFun := ⇑(LinearMap.reduceModIdealAux I f), map_add' := ⋯ }.toFun (HS …
    -/
    refine Quotient.inductionOn' x (fun x ↦ ?_)
    simp only [Submodule.Quotient.mk''_eq_mk, Ideal.Quotient.mk_eq_mk, Module.Quotient.mk_smul_mk,
      Submodule.Quotient.mk_smul, LinearMapClass.map_smul, reduceModIdealAux_apply,
      RingHomCompTriple.comp_apply]


@[simp]
theorem reduceModIdeal_apply (f : M →ₗ[R] N) (x : M) :
    (f.reduceModIdeal I) (Submodule.Quotient.mk (p := (I • ⊤ : Submodule R M)) x) =
      Submodule.Quotient.mk (p := (I • ⊤ : Submodule R N)) (f x) :=
  rfl


theorem transitionMap_comp_reduceModIdeal (f : M →ₗ[R] N) {m n : ℕ}
    (hmn : m ≤ n) : transitionMap I N hmn ∘ₗ f.reduceModIdeal (I ^ n) =
      (f.reduceModIdeal (I ^ m) : _ →ₗ[R] _) ∘ₗ transitionMap I M hmn := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq ((AdicCompletion.transitionMap I N hmn).comp (↑R (LinearMap.reduceModIdea …
  -/
  ext x
  /-
    case h.h
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    m n : Nat
    hmn : LE.le m n
    x : M
    ⊢ Eq ((((AdicCompletion.transitionMap I N hmn).comp (↑R (LinearMap.reduceModId …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A linear map induces a linear map on adic cauchy sequences. -/
@[simps]
def map (f : M →ₗ[R] N) : AdicCauchySequence I M →ₗ[R] AdicCauchySequence I N where
  toFun a := ⟨fun n ↦ f (a n), fun {m n} hmn ↦ by
    have hm : Submodule.map f (I ^ m • ⊤ : Submodule R M) ≤ (I ^ m • ⊤ : Submodule R N) := by
      rw [Submodule.map_smul'']
      exact smul_mono_right _ le_top
    /-
      R : Type u_1
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u_3
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u_4
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      T : Type u_5
      inst✝¹ : AddCommGroup T
      inst✝ : Module (AdicCompletion I R) T
      f : LinearMap (RingHom.id R) M N
      a : AdicCompletion.AdicCauchySequence I M
      m n : Nat
      hmn : LE.le m n
      hm : LE.le (Submodule.map f (HSMul.hSMul (HPow.hPow I m) Top.top)) (HSMul.hSMu …
      ⊢ SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ((fun n => f (↑a n)) m) ((fun n …
    -/
    apply SModEq.mono hm
    /-
      R : Type u_1
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u_3
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u_4
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      T : Type u_5
      inst✝¹ : AddCommGroup T
      inst✝ : Module (AdicCompletion I R) T
      f : LinearMap (RingHom.id R) M N
      a : AdicCompletion.AdicCauchySequence I M
      m n : Nat
      hmn : LE.le m n
      hm : LE.le (Submodule.map f (HSMul.hSMul (HPow.hPow I m) Top.top)) (HSMul.hSMu …
      ⊢ SModEq (Submodule.map f (HSMul.hSMul (HPow.hPow I m) Top.top)) ((fun n => f  …
    -/
    apply SModEq.map (a.property hmn) f⟩
    /-
      🎉 no goals
    -/
                     /-
                       R : Type u_1
                       inst✝⁸ : CommRing R
                       I : Ideal R
                       M : Type u_2
                       inst✝⁷ : AddCommGroup M
                       inst✝⁶ : Module R M
                       N : Type u_3
                       inst✝⁵ : AddCommGroup N
                       inst✝⁴ : Module R N
                       P : Type u_4
                       inst✝³ : AddCommGroup P
                       inst✝² : Module R P
                       T : Type u_5
                       inst✝¹ : AddCommGroup T
                       inst✝ : Module (AdicCompletion I R) T
                       f : LinearMap (RingHom.id R) M N
                       a b : AdicCompletion.AdicCauchySequence I M
                       ⊢ Eq ((fun a => ⟨fun n => f (↑a n), ⋯⟩) (HAdd.hAdd a b)) (HAdd.hAdd ((fun a => …
                     -/
  map_add' a b := by ext n; simp
                            /-
                              🎉 no goals
                            -/
                      /-
                        R : Type u_1
                        inst✝⁸ : CommRing R
                        I : Ideal R
                        M : Type u_2
                        inst✝⁷ : AddCommGroup M
                        inst✝⁶ : Module R M
                        N : Type u_3
                        inst✝⁵ : AddCommGroup N
                        inst✝⁴ : Module R N
                        P : Type u_4
                        inst✝³ : AddCommGroup P
                        inst✝² : Module R P
                        T : Type u_5
                        inst✝¹ : AddCommGroup T
                        inst✝ : Module (AdicCompletion I R) T
                        f : LinearMap (RingHom.id R) M N
                        r : R
                        a : AdicCompletion.AdicCauchySequence I M
                        ⊢ Eq ({ toFun := fun a => ⟨fun n => f (↑a n), ⋯⟩, map_add' := ⋯ }.toFun (HSMul …
                      -/
  map_smul' r a := by ext n; simp
                             /-
                               🎉 no goals
                             -/


variable (M) in
@[simp]
theorem map_id : map I (LinearMap.id (M := M)) = LinearMap.id :=
  rfl


theorem map_comp (f : M →ₗ[R] N) (g : N →ₗ[R] P) :
    map I g ∘ₗ map I f = map I (g ∘ₗ f) :=
  rfl


theorem map_comp_apply (f : M →ₗ[R] N) (g : N →ₗ[R] P) (a : AdicCauchySequence I M) :
    map I g (map I f a) = map I (g ∘ₗ f) a :=
  rfl


@[simp]
theorem map_zero : map I (0 : M →ₗ[R] N) = 0 :=
  rfl


/-- `R`-linear version of `adicCompletion`. -/
private def adicCompletionAux (f : M →ₗ[R] N) :
    AdicCompletion I M →ₗ[R] AdicCompletion I N :=
  AdicCompletion.lift I (fun n ↦ reduceModIdeal (I ^ n) f ∘ₗ AdicCompletion.eval I M n)
    (fun {m n} hmn ↦ by rw [← comp_assoc, AdicCompletion.transitionMap_comp_reduceModIdeal,
        comp_assoc, transitionMap_comp_eval])


@[local simp]
private theorem adicCompletionAux_val_apply (f : M →ₗ[R] N) {n : ℕ} (x : AdicCompletion I M) :
    (adicCompletionAux I f x).val n = f.reduceModIdeal (I ^ n) (x.val n) :=
  rfl


/-- A linear map induces a map on adic completions. -/
def map (f : M →ₗ[R] N) :
    AdicCompletion I M →ₗ[AdicCompletion I R] AdicCompletion I N where
  toFun := adicCompletionAux I f
                 /-
                   R : Type u_1
                   inst✝⁸ : CommRing R
                   I : Ideal R
                   M : Type u_2
                   inst✝⁷ : AddCommGroup M
                   inst✝⁶ : Module R M
                   N : Type u_3
                   inst✝⁵ : AddCommGroup N
                   inst✝⁴ : Module R N
                   P : Type u_4
                   inst✝³ : AddCommGroup P
                   inst✝² : Module R P
                   T : Type u_5
                   inst✝¹ : AddCommGroup T
                   inst✝ : Module (AdicCompletion I R) T
                   f : LinearMap (RingHom.id R) M N
                   ⊢ ∀ (x y : AdicCompletion I M), Eq ((AdicCompletion.adicCompletionAux I f) (HA …
                 -/
  map_add' := by aesop
                 /-
                   🎉 no goals
                 -/
  map_smul' r x := by
    /-
      R : Type u_1
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u_3
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u_4
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      T : Type u_5
      inst✝¹ : AddCommGroup T
      inst✝ : Module (AdicCompletion I R) T
      f : LinearMap (RingHom.id R) M N
      r : AdicCompletion I R
      x : AdicCompletion I M
      ⊢ Eq ({ toFun := ⇑(AdicCompletion.adicCompletionAux I f), map_add' := ⋯ }.toFu …
    -/
    ext n
    /-
      case h
      R : Type u_1
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u_3
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u_4
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      T : Type u_5
      inst✝¹ : AddCommGroup T
      inst✝ : Module (AdicCompletion I R) T
      f : LinearMap (RingHom.id R) M N
      r : AdicCompletion I R
      x : AdicCompletion I M
      n : Nat
      ⊢ Eq (↑({ toFun := ⇑(AdicCompletion.adicCompletionAux I f), map_add' := ⋯ }.to …
    -/
    simp only [adicCompletionAux_val_apply, smul_eval, smul_eq_mul, RingHom.id_apply]
    /-
      case h
      R : Type u_1
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u_3
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u_4
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      T : Type u_5
      inst✝¹ : AddCommGroup T
      inst✝ : Module (AdicCompletion I R) T
      f : LinearMap (RingHom.id R) M N
      r : AdicCompletion I R
      x : AdicCompletion I M
      n : Nat
      ⊢ Eq ((LinearMap.reduceModIdeal (HPow.hPow I n) f) (HSMul.hSMul (↑r n) (↑x n)) …
    -/
    rw [val_smul_eq_evalₐ_smul, val_smul_eq_evalₐ_smul, map_smul]
    /-
      🎉 no goals
    -/


@[simp]
theorem map_val_apply (f : M →ₗ[R] N) {n : ℕ} (x : AdicCompletion I M) :
    (map I f x).val n = f.reduceModIdeal (I ^ n) (x.val n) :=
  rfl


/-- Equality of maps out of an adic completion can be checked on Cauchy sequences. -/
theorem map_ext {N} {f g : AdicCompletion I M → N}
    (h : ∀ (a : AdicCauchySequence I M),
      f (AdicCompletion.mk I M a) = g (AdicCompletion.mk I M a)) :
    f = g := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    N : Sort u_6
    f g : AdicCompletion I M → N
    h : ∀ (a : AdicCompletion.AdicCauchySequence I M), Eq (f ((AdicCompletion.mk I …
    ⊢ Eq f g
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    N : Sort u_6
    f g : AdicCompletion I M → N
    h : ∀ (a : AdicCompletion.AdicCauchySequence I M), Eq (f ((AdicCompletion.mk I …
    x : AdicCompletion I M
    ⊢ Eq (f x) (g x)
  -/
  apply induction_on I M x h
  /-
    🎉 no goals
  -/


/-- Equality of linear maps out of an adic completion can be checked on Cauchy sequences. -/
@[ext]
theorem map_ext' {f g : AdicCompletion I M →ₗ[AdicCompletion I R] T}
    (h : ∀ (a : AdicCauchySequence I M),
      f (AdicCompletion.mk I M a) = g (AdicCompletion.mk I M a)) :
    f = g := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    T : Type u_5
    inst✝¹ : AddCommGroup T
    inst✝ : Module (AdicCompletion I R) T
    f g : LinearMap (RingHom.id (AdicCompletion I R)) (AdicCompletion I M) T
    h : ∀ (a : AdicCompletion.AdicCauchySequence I M), Eq (f ((AdicCompletion.mk I …
    ⊢ Eq f g
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    T : Type u_5
    inst✝¹ : AddCommGroup T
    inst✝ : Module (AdicCompletion I R) T
    f g : LinearMap (RingHom.id (AdicCompletion I R)) (AdicCompletion I M) T
    h : ∀ (a : AdicCompletion.AdicCauchySequence I M), Eq (f ((AdicCompletion.mk I …
    x : AdicCompletion I M
    ⊢ Eq (f x) (g x)
  -/
  apply induction_on I M x h
  /-
    🎉 no goals
  -/


/-- Equality of linear maps out of an adic completion can be checked on Cauchy sequences. -/
@[ext]
theorem map_ext'' {f g : AdicCompletion I M →ₗ[R] N}
    (h : f.comp (AdicCompletion.mk I M) = g.comp (AdicCompletion.mk I M)) :
    f = g := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f g : LinearMap (RingHom.id R) (AdicCompletion I M) N
    h : Eq (f.comp (AdicCompletion.mk I M)) (g.comp (AdicCompletion.mk I M))
    ⊢ Eq f g
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f g : LinearMap (RingHom.id R) (AdicCompletion I M) N
    h : Eq (f.comp (AdicCompletion.mk I M)) (g.comp (AdicCompletion.mk I M))
    x : AdicCompletion I M
    ⊢ Eq (f x) (g x)
  -/
  apply induction_on I M x (fun a ↦ LinearMap.ext_iff.mp h a)
  /-
    🎉 no goals
  -/


variable (M) in
@[simp]
theorem map_id :
    map I (LinearMap.id (M := M)) =
      LinearMap.id (R := AdicCompletion I R) (M := AdicCompletion I M) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (AdicCompletion.map I LinearMap.id) LinearMap.id
  -/
  ext a n
  /-
    case h.h
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    a : AdicCompletion.AdicCauchySequence I M
    n : Nat
    ⊢ Eq (↑((AdicCompletion.map I LinearMap.id) ((AdicCompletion.mk I M) a)) n) (↑ …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem map_comp (f : M →ₗ[R] N) (g : N →ₗ[R] P) :
    map I g ∘ₗ map I f = map I (g ∘ₗ f) := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    P : Type u_4
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    ⊢ Eq ((AdicCompletion.map I g).comp (AdicCompletion.map I f)) (AdicCompletion. …
  -/
  ext
  /-
    case h.h
    R : Type u_1
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    P : Type u_4
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    a✝ : AdicCompletion.AdicCauchySequence I M
    n✝ : Nat
    ⊢ Eq (↑(((AdicCompletion.map I g).comp (AdicCompletion.map I f)) ((AdicComplet …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem map_comp_apply (f : M →ₗ[R] N) (g : N →ₗ[R] P) (x : AdicCompletion I M) :
    map I g (map I f x) = map I (g ∘ₗ f) x := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    P : Type u_4
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    x : AdicCompletion I M
    ⊢ Eq ((AdicCompletion.map I g) ((AdicCompletion.map I f) x)) ((AdicCompletion. …
  -/
  show (map I g ∘ₗ map I f) x = map I (g ∘ₗ f) x
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    P : Type u_4
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    x : AdicCompletion I M
    ⊢ Eq (((AdicCompletion.map I g).comp (AdicCompletion.map I f)) x) ((AdicComple …
  -/
  rw [map_comp]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_mk (f : M →ₗ[R] N) (a : AdicCauchySequence I M) :
    map I f (AdicCompletion.mk I M a) =
      AdicCompletion.mk I N (AdicCauchySequence.map I f a) :=
  rfl


@[simp]
theorem map_zero : map I (0 : M →ₗ[R] N) = 0 := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    ⊢ Eq (AdicCompletion.map I 0) 0
  -/
  ext
  /-
    case h.h
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    a✝ : AdicCompletion.AdicCauchySequence I M
    n✝ : Nat
    ⊢ Eq (↑((AdicCompletion.map I 0) ((AdicCompletion.mk I M) a✝)) n✝) (↑(0 ((Adic …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A linear equiv induces a linear equiv on adic completions. -/
def congr (f : M ≃ₗ[R] N) :
    AdicCompletion I M ≃ₗ[AdicCompletion I R] AdicCompletion I N :=
  LinearEquiv.ofLinear (map I f)
                       /-
                         R : Type u_1
                         inst✝⁸ : CommRing R
                         I : Ideal R
                         M : Type u_2
                         inst✝⁷ : AddCommGroup M
                         inst✝⁶ : Module R M
                         N : Type u_3
                         inst✝⁵ : AddCommGroup N
                         inst✝⁴ : Module R N
                         P : Type u_4
                         inst✝³ : AddCommGroup P
                         inst✝² : Module R P
                         T : Type u_5
                         inst✝¹ : AddCommGroup T
                         inst✝ : Module (AdicCompletion I R) T
                         f : LinearEquiv (RingHom.id R) M N
                         ⊢ Eq ((AdicCompletion.map I ↑f).comp (AdicCompletion.map I ↑f.symm)) LinearMap …
                       -/
                       /-
                         🎉 no goals
                       -/
    (map I f.symm) (by simp [map_comp]) (by simp [map_comp])
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem congr_apply (f : M ≃ₗ[R] N) (x : AdicCompletion I M) :
    congr I f x = map I f x :=
  rfl


@[simp]
theorem congr_symm_apply (f : M ≃ₗ[R] N) (x : AdicCompletion I N) :
    (congr I f).symm x = map I f.symm x :=
  rfl


/-- The canonical map from the adic completion of the product to the product of the
adic completions. -/
@[simps!]
def pi : AdicCompletion I (∀ j, M j) →ₗ[AdicCompletion I R] ∀ j, AdicCompletion I (M j) :=
  LinearMap.pi (fun j ↦ map I (LinearMap.proj j))


/-- The canonical map from the sum of the adic completions to the adic completion
of the sum. -/
def sum [DecidableEq ι] :
    (⨁ j, (AdicCompletion I (M j))) →ₗ[AdicCompletion I R] AdicCompletion I (⨁ j, M j) :=
  toModule (AdicCompletion I R) ι (AdicCompletion I (⨁ j, M j))
    (fun j ↦ map I (lof R ι M j))


@[simp]
theorem sum_lof [DecidableEq ι] (j : ι) (x : AdicCompletion I (M j)) :
    sum I M ((DirectSum.lof (AdicCompletion I R) ι (fun i ↦ AdicCompletion I (M i)) j) x) =
      map I (lof R ι M j) x := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝² : (i : ι) → AddCommGroup (M i)
    inst✝¹ : (i : ι) → Module R (M i)
    inst✝ : DecidableEq ι
    j : ι
    x : AdicCompletion I (M j)
    ⊢ Eq ((AdicCompletion.sum I M) ((DirectSum.lof (AdicCompletion I R) ι (fun i = …
  -/
  simp [sum]
  /-
    🎉 no goals
  -/


@[simp]
theorem sum_of [DecidableEq ι] (j : ι) (x : AdicCompletion I (M j)) :
    sum I M ((DirectSum.of (fun i ↦ AdicCompletion I (M i)) j) x) =
      map I (lof R ι M j) x := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝² : (i : ι) → AddCommGroup (M i)
    inst✝¹ : (i : ι) → Module R (M i)
    inst✝ : DecidableEq ι
    j : ι
    x : AdicCompletion I (M j)
    ⊢ Eq ((AdicCompletion.sum I M) ((DirectSum.of (fun i => AdicCompletion I (M i) …
  -/
  rw [← lof_eq_of R]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝² : (i : ι) → AddCommGroup (M i)
    inst✝¹ : (i : ι) → Module R (M i)
    inst✝ : DecidableEq ι
    j : ι
    x : AdicCompletion I (M j)
    ⊢ Eq ((AdicCompletion.sum I M) ((DirectSum.lof R ι (fun i => AdicCompletion I  …
  -/
  apply sum_lof
  /-
    🎉 no goals
  -/


/-- If `ι` is finite, we use the equivalence of sum and product to obtain an inverse for
`AdicCompletion.sum` from `AdicCompletion.pi`. -/
def sumInv : AdicCompletion I (⨁ j, M j) →ₗ[AdicCompletion I R] (⨁ j, (AdicCompletion I (M j))) :=
  letI f := map I (linearEquivFunOnFintype R ι M)
  letI g := linearEquivFunOnFintype (AdicCompletion I R) ι (fun j ↦ AdicCompletion I (M j))
  g.symm.toLinearMap ∘ₗ pi I M ∘ₗ f


@[simp]
theorem component_sumInv (x : AdicCompletion I (⨁ j, M j)) (j : ι) :
    component (AdicCompletion I R) ι _ j (sumInv I M x) =
      map I (component R ι _ j) x := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝² : (i : ι) → AddCommGroup (M i)
    inst✝¹ : (i : ι) → Module R (M i)
    inst✝ : Fintype ι
    x : AdicCompletion I (DirectSum ι fun j => M j)
    j : ι
    ⊢ Eq ((DirectSum.component (AdicCompletion I R) ι (fun i => AdicCompletion I ( …
  -/
  apply induction_on I _ x (fun x ↦ ?_)
  /-
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝² : (i : ι) → AddCommGroup (M i)
    inst✝¹ : (i : ι) → Module R (M i)
    inst✝ : Fintype ι
    x✝ : AdicCompletion I (DirectSum ι fun j => M j)
    j : ι
    x : AdicCompletion.AdicCauchySequence I (DirectSum ι fun j => M j)
    ⊢ Eq ((DirectSum.component (AdicCompletion I R) ι (fun i => AdicCompletion I ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem sumInv_apply (x : AdicCompletion I (⨁ j, M j)) (j : ι) :
    (sumInv I M x) j = map I (component R ι _ j) x := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝² : (i : ι) → AddCommGroup (M i)
    inst✝¹ : (i : ι) → Module R (M i)
    inst✝ : Fintype ι
    x : AdicCompletion I (DirectSum ι fun j => M j)
    j : ι
    ⊢ Eq (((AdicCompletion.sumInv I M) x) j) ((AdicCompletion.map I (DirectSum.com …
  -/
  apply induction_on I _ x (fun x ↦ ?_)
  /-
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝² : (i : ι) → AddCommGroup (M i)
    inst✝¹ : (i : ι) → Module R (M i)
    inst✝ : Fintype ι
    x✝ : AdicCompletion I (DirectSum ι fun j => M j)
    j : ι
    x : AdicCompletion.AdicCauchySequence I (DirectSum ι fun j => M j)
    ⊢ Eq (((AdicCompletion.sumInv I M) ((AdicCompletion.mk I (DirectSum ι fun j => …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem sumInv_comp_sum : sumInv I M ∘ₗ sum I M = LinearMap.id := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((AdicCompletion.sumInv I M).comp (AdicCompletion.sum I M)) LinearMap.id
  -/
  ext j x
  /-
    case H.h
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    j : ι
    x : AdicCompletion.AdicCauchySequence I (M j)
    ⊢ Eq ((((AdicCompletion.sumInv I M).comp (AdicCompletion.sum I M)).comp (Direc …
  -/
  apply DirectSum.ext (AdicCompletion I R) (fun i ↦ ?_)
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    j : ι
    x : AdicCompletion.AdicCauchySequence I (M j)
    i : ι
    ⊢ Eq ((DirectSum.component (AdicCompletion I R) ι (fun i => AdicCompletion I ( …
  -/
  ext n
  simp only [LinearMap.coe_comp, Function.comp_apply, sum_lof, map_mk, component_sumInv,
    mk_apply_coe, AdicCauchySequence.map_apply_coe, Submodule.mkQ_apply, LinearMap.id_comp]
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    j : ι
    x : AdicCompletion.AdicCauchySequence I (M j)
    i : ι
    n : Nat
    ⊢ Eq (Submodule.Quotient.mk ((DirectSum.component R ι M i) ((DirectSum.lof R ι …
  -/
  rw [DirectSum.component.of, DirectSum.component.of]
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    j : ι
    x : AdicCompletion.AdicCauchySequence I (M j)
    i : ι
    n : Nat
    ⊢ Eq (Submodule.Quotient.mk (dite (Eq j i) (fun h => Eq.recOn h (↑x n)) fun h  …
  -/
  split
    /-
      case h.isTrue
      R : Type u_1
      inst✝⁴ : CommRing R
      I : Ideal R
      ι : Type u_6
      M : ι → Type u_7
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      j : ι
      x : AdicCompletion.AdicCauchySequence I (M j)
      i : ι
      n : Nat
      h✝ : Eq j i
      ⊢ Eq (Submodule.Quotient.mk (Eq.recOn h✝ (↑x n))) (↑(Eq.recOn h✝ ((AdicComplet …
    -/
  · next h => subst h; simp
    /-
      🎉 no goals
    -/
    /-
      case h.isFalse
      R : Type u_1
      inst✝⁴ : CommRing R
      I : Ideal R
      ι : Type u_6
      M : ι → Type u_7
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      j : ι
      x : AdicCompletion.AdicCauchySequence I (M j)
      i : ι
      n : Nat
      h✝ : Not (Eq j i)
      ⊢ Eq (Submodule.Quotient.mk 0) (↑0 n)
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem sum_comp_sumInv : sum I M ∘ₗ sumInv I M = LinearMap.id := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((AdicCompletion.sum I M).comp (AdicCompletion.sumInv I M)) LinearMap.id
  -/
  ext f n
  simp only [LinearMap.coe_comp, Function.comp_apply, LinearMap.id_coe, id_eq, mk_apply_coe,
    Submodule.mkQ_apply]
  /-
    case h.h
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : AdicCompletion.AdicCauchySequence I (DirectSum ι fun j => M j)
    n : Nat
    ⊢ Eq (↑((AdicCompletion.sum I M) ((AdicCompletion.sumInv I M) ((AdicCompletion …
  -/
  rw [← DirectSum.sum_univ_of (((sumInv I M) ((AdicCompletion.mk I (⨁ (j : ι), M j)) f)))]
  simp only [sumInv_apply, map_mk, map_sum, sum_of, val_sum_apply, mk_apply_coe,
    AdicCauchySequence.map_apply_coe, Submodule.mkQ_apply]
  /-
    case h.h
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : AdicCompletion.AdicCauchySequence I (DirectSum ι fun j => M j)
    n : Nat
    ⊢ Eq (Finset.univ.sum fun x => Submodule.Quotient.mk ((DirectSum.lof R ι M x)  …
  -/
  simp only [← Submodule.mkQ_apply, ← map_sum]
  /-
    case h.h
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : AdicCompletion.AdicCauchySequence I (DirectSum ι fun j => M j)
    n : Nat
    ⊢ Eq ((HSMul.hSMul (HPow.hPow I n) Top.top).mkQ (Finset.univ.sum fun x => (Dir …
  -/
  erw [DirectSum.sum_univ_of]
  /-
    🎉 no goals
  -/


/-- If `ι` is finite, `sum` has `sumInv` as inverse. -/
def sumEquivOfFintype :
    (⨁ j, (AdicCompletion I (M j))) ≃ₗ[AdicCompletion I R] AdicCompletion I (⨁ j, M j) :=
  LinearEquiv.ofLinear (sum I M) (sumInv I M) (sum_comp_sumInv I M) (sumInv_comp_sum I M)


@[simp]
theorem sumEquivOfFintype_apply (x : ⨁ j, (AdicCompletion I (M j))) :
    sumEquivOfFintype I M x = sum I M x :=
  rfl


@[simp]
theorem sumEquivOfFintype_symm_apply (x : AdicCompletion I (⨁ j, M j)) :
    (sumEquivOfFintype I M).symm x = sumInv I M x :=
  rfl


/-- If `ι` is finite, `pi` is a linear equiv. -/
def piEquivOfFintype :
    AdicCompletion I (∀ j, M j) ≃ₗ[AdicCompletion I R] ∀ j, AdicCompletion I (M j) :=
  letI f := (congr I (linearEquivFunOnFintype R ι M)).symm
  letI g := (linearEquivFunOnFintype (AdicCompletion I R) ι (fun j ↦ AdicCompletion I (M j)))
  f.trans ((sumEquivOfFintype I M).symm.trans g)


@[simp]
theorem piEquivOfFintype_apply (x : AdicCompletion I (∀ j, M j)) :
    piEquivOfFintype I M x = pi I M x := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    ι : Type u_6
    M : ι → Type u_7
    inst✝³ : (i : ι) → AddCommGroup (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    x : AdicCompletion I ((j : ι) → M j)
    ⊢ Eq ((AdicCompletion.piEquivOfFintype I M) x) ((AdicCompletion.pi I M) x)
  -/
  simp [piEquivOfFintype, sumInv, map_comp_apply]
  /-
    🎉 no goals
  -/


/-- Adic completion of `R^n` is `(AdicCompletion I R)^n`. -/
def piEquivFin (n : ℕ) :
    AdicCompletion I (Fin n → R) ≃ₗ[AdicCompletion I R] Fin n → AdicCompletion I R :=
  piEquivOfFintype I (ι := Fin n) (fun _ : Fin n ↦ R)


@[simp]
theorem piEquivFin_apply (n : ℕ) (x : AdicCompletion I (Fin n → R)) :
    piEquivFin I n x = pi I (fun _ : Fin n ↦ R) x := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    n : Nat
    x : AdicCompletion I (Fin n → R)
    ⊢ Eq ((AdicCompletion.piEquivFin I n) x) ((AdicCompletion.pi I fun x => R) x)
  -/
  simp only [piEquivFin, piEquivOfFintype_apply]
  /-
    🎉 no goals
  -/


