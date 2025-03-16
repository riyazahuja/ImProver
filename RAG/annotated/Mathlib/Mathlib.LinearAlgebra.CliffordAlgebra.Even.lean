/-- The even submodule `CliffordAlgebra.evenOdd Q 0` is also a subalgebra. -/
def even : Subalgebra R (CliffordAlgebra Q) :=
  (evenOdd Q 0).toSubalgebra (SetLike.one_mem_graded _) fun _x _y hx hy =>
    add_zero (0 : ZMod 2) ▸ SetLike.mul_mem_graded hx hy

-- Porting note: added, otherwise Lean can't find this when it needs it

instance : AddCommMonoid (even Q) := AddSubmonoidClass.toAddCommMonoid _

@[simp]
theorem even_toSubmodule : Subalgebra.toSubmodule (even Q) = evenOdd Q 0 :=
  rfl


/-- The type of bilinear maps which are accepted by `CliffordAlgebra.even.lift`. -/
@[ext]
structure EvenHom : Type max uA uM where
  bilin : M →ₗ[R] M →ₗ[R] A
  contract (m : M) : bilin m m = algebraMap R A (Q m)
  contract_mid (m₁ m₂ m₃ : M) : bilin m₁ m₂ * bilin m₂ m₃ = Q m₂ • bilin m₁ m₃


/-- Compose an `EvenHom` with an `AlgHom` on the output. -/
@[simps]
def EvenHom.compr₂ (g : EvenHom Q A) (f : A →ₐ[R] B) : EvenHom Q B where
  bilin := g.bilin.compr₂ f.toLinearMap
  contract _m := (f.congr_arg <| g.contract _).trans <| f.commutes _
  contract_mid _m₁ _m₂ _m₃ :=
    (map_mul f _ _).symm.trans <| (f.congr_arg <| g.contract_mid _ _ _).trans <| map_smul f _ _


/-- The embedding of pairs of vectors into the even subalgebra, as a bilinear map. -/
nonrec def even.ι : EvenHom Q (even Q) where
  bilin :=
    LinearMap.mk₂ R (fun m₁ m₂ => ⟨ι Q m₁ * ι Q m₂, ι_mul_ι_mem_evenOdd_zero Q _ _⟩)
                       /-
                         R : Type uR
                         M : Type uM
                         inst✝⁶ : CommRing R
                         inst✝⁵ : AddCommGroup M
                         inst✝⁴ : Module R M
                         Q : QuadraticForm R M
                         A : Type uA
                         B : Type uB
                         inst✝³ : Ring A
                         inst✝² : Ring B
                         inst✝¹ : Algebra R A
                         inst✝ : Algebra R B
                         x✝² x✝¹ x✝ : M
                         ⊢ Eq ((fun m₁ m₂ => ⟨HMul.hMul ((CliffordAlgebra.ι Q) m₁) ((CliffordAlgebra.ι  …
                       -/
      (fun _ _ _ => by simp only [LinearMap.map_add, add_mul]; rfl)
                                                               /-
                                                                 🎉 no goals
                                                               -/
                       /-
                         R : Type uR
                         M : Type uM
                         inst✝⁶ : CommRing R
                         inst✝⁵ : AddCommGroup M
                         inst✝⁴ : Module R M
                         Q : QuadraticForm R M
                         A : Type uA
                         B : Type uB
                         inst✝³ : Ring A
                         inst✝² : Ring B
                         inst✝¹ : Algebra R A
                         inst✝ : Algebra R B
                         x✝² : R
                         x✝¹ x✝ : M
                         ⊢ Eq ((fun m₁ m₂ => ⟨HMul.hMul ((CliffordAlgebra.ι Q) m₁) ((CliffordAlgebra.ι  …
                       -/
      (fun _ _ _ => by simp only [LinearMap.map_smul, smul_mul_assoc]; rfl)
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                       /-
                         R : Type uR
                         M : Type uM
                         inst✝⁶ : CommRing R
                         inst✝⁵ : AddCommGroup M
                         inst✝⁴ : Module R M
                         Q : QuadraticForm R M
                         A : Type uA
                         B : Type uB
                         inst✝³ : Ring A
                         inst✝² : Ring B
                         inst✝¹ : Algebra R A
                         inst✝ : Algebra R B
                         x✝² x✝¹ x✝ : M
                         ⊢ Eq ((fun m₁ m₂ => ⟨HMul.hMul ((CliffordAlgebra.ι Q) m₁) ((CliffordAlgebra.ι  …
                       -/
      (fun _ _ _ => by simp only [LinearMap.map_add, mul_add]; rfl) fun _ _ _ => by
                                                               /-
                                                                 🎉 no goals
                                                               -/
      /-
        R : Type uR
        M : Type uM
        inst✝⁶ : CommRing R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        Q : QuadraticForm R M
        A : Type uA
        B : Type uB
        inst✝³ : Ring A
        inst✝² : Ring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        x✝² : R
        x✝¹ x✝ : M
        ⊢ Eq ((fun m₁ m₂ => ⟨HMul.hMul ((CliffordAlgebra.ι Q) m₁) ((CliffordAlgebra.ι  …
      -/
      simp only [LinearMap.map_smul, mul_smul_comm]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/
  contract m := Subtype.ext <| ι_sq_scalar Q m
  contract_mid m₁ m₂ m₃ :=
    Subtype.ext <|
      calc
        ι Q m₁ * ι Q m₂ * (ι Q m₂ * ι Q m₃) = ι Q m₁ * (ι Q m₂ * ι Q m₂ * ι Q m₃) := by
          /-
            R : Type uR
            M : Type uM
            inst✝⁶ : CommRing R
            inst✝⁵ : AddCommGroup M
            inst✝⁴ : Module R M
            Q : QuadraticForm R M
            A : Type uA
            B : Type uB
            inst✝³ : Ring A
            inst✝² : Ring B
            inst✝¹ : Algebra R A
            inst✝ : Algebra R B
            m₁ m₂ m₃ : M
            ⊢ Eq (HMul.hMul (HMul.hMul ((CliffordAlgebra.ι Q) m₁) ((CliffordAlgebra.ι Q) m …
          -/
          simp only [mul_assoc]
          /-
            🎉 no goals
          -/
                                           /-
                                             R : Type uR
                                             M : Type uM
                                             inst✝⁶ : CommRing R
                                             inst✝⁵ : AddCommGroup M
                                             inst✝⁴ : Module R M
                                             Q : QuadraticForm R M
                                             A : Type uA
                                             B : Type uB
                                             inst✝³ : Ring A
                                             inst✝² : Ring B
                                             inst✝¹ : Algebra R A
                                             inst✝ : Algebra R B
                                             m₁ m₂ m₃ : M
                                             ⊢ Eq (HMul.hMul ((CliffordAlgebra.ι Q) m₁) (HMul.hMul (HMul.hMul ((CliffordAlg …
                                           -/
        _ = Q m₂ • (ι Q m₁ * ι Q m₃) := by rw [Algebra.smul_def, ι_sq_scalar, Algebra.left_comm]
                                           /-
                                             🎉 no goals
                                           -/


instance : Inhabited (EvenHom Q (even Q)) :=
  ⟨even.ι Q⟩


/-- Two algebra morphisms from the even subalgebra are equal if they agree on pairs of generators.

See note [partially-applied ext lemmas]. -/
@[ext high]
theorem even.algHom_ext ⦃f g : even Q →ₐ[R] A⦄ (h : (even.ι Q).compr₂ f = (even.ι Q).compr₂ g) :
    f = g := by
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type uA
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    f g : AlgHom R (Subtype fun x => Membership.mem (CliffordAlgebra.even Q) x) A
    h : Eq ((CliffordAlgebra.even.ι Q).compr₂ f) ((CliffordAlgebra.even.ι Q).compr …
    ⊢ Eq f g
  -/
  rw [EvenHom.ext_iff] at h
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type uA
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    f g : AlgHom R (Subtype fun x => Membership.mem (CliffordAlgebra.even Q) x) A
    h : Eq ((CliffordAlgebra.even.ι Q).compr₂ f).bilin ((CliffordAlgebra.even.ι Q) …
    ⊢ Eq f g
  -/
  ext ⟨x, hx⟩
  induction x, hx using even_induction with
  | algebraMap r =>
    exact (f.commutes r).trans (g.commutes r).symm
  | add x y hx hy ihx ihy =>
    have := congr_arg₂ (· + ·) ihx ihy
    exact (map_add f _ _).trans (this.trans <| (map_add g _ _).symm)
  | ι_mul_ι_mul m₁ m₂ x hx ih =>
    have := congr_arg₂ (· * ·) (LinearMap.congr_fun (LinearMap.congr_fun h m₁) m₂) ih
    exact (map_mul f _ _).trans (this.trans <| (map_mul g _ _).symm)


/-- An auxiliary submodule used to store the half-applied values of `f`.
This is the span of elements `f'` such that `∃ x m₂, ∀ m₁, f' m₁ = f m₁ m₂ * x`. -/
private def S : Submodule R (M →ₗ[R] A) :=
  Submodule.span R
    {f' | ∃ x m₂, f' = LinearMap.lcomp R _ (f.bilin.flip m₂) (LinearMap.mulRight R x)}


/-- An auxiliary bilinear map that is later passed into `CliffordAlgebra.foldr`. Our desired result
is stored in the `A` part of the accumulator, while auxiliary recursion state is stored in the `S f`
part. -/
private def fFold : M →ₗ[R] A × S f →ₗ[R] A × S f :=
  LinearMap.mk₂ R
    (fun m acc =>
      /- We could write this `snd` term in a point-free style as follows, but it wouldn't help as we
        don't have any prod or subtype combinators to deal with n-linear maps of this degree.
        ```lean
        (LinearMap.lcomp R _ (Algebra.lmul R A).to_linear_map.flip).comp <|
          (LinearMap.llcomp R M A A).flip.comp f.flip : M →ₗ[R] A →ₗ[R] M →ₗ[R] A)
        ```
        -/
      (acc.2.val m,
        ⟨(LinearMap.mulRight R acc.1).comp (f.bilin.flip m), Submodule.subset_span <| ⟨_, _, rfl⟩⟩))
    (fun m₁ m₂ a =>
      Prod.ext (LinearMap.map_add _ m₁ m₂)
        (Subtype.ext <|
          LinearMap.ext fun m₃ =>
            show f.bilin m₃ (m₁ + m₂) * a.1 = f.bilin m₃ m₁ * a.1 + f.bilin m₃ m₂ * a.1 by
              /-
                R : Type uR
                M : Type uM
                inst✝⁶ : CommRing R
                inst✝⁵ : AddCommGroup M
                inst✝⁴ : Module R M
                Q : QuadraticForm R M
                A : Type uA
                B : Type uB
                inst✝³ : Ring A
                inst✝² : Ring B
                inst✝¹ : Algebra R A
                inst✝ : Algebra R B
                f : CliffordAlgebra.EvenHom Q A
                m₁ m₂ : M
                a : Prod A (Subtype fun x => Membership.mem (CliffordAlgebra.even.lift.S f) x)
                m₃ : M
                ⊢ Eq (HMul.hMul ((f.bilin m₃) (HAdd.hAdd m₁ m₂)) a.1) (HAdd.hAdd (HMul.hMul (( …
              -/
              rw [map_add, add_mul]))
              /-
                🎉 no goals
              -/
    (fun c m a =>
      Prod.ext (LinearMap.map_smul _ c m)
        (Subtype.ext <|
          LinearMap.ext fun m₃ =>
            show f.bilin m₃ (c • m) * a.1 = c • (f.bilin m₃ m * a.1) by
              /-
                R : Type uR
                M : Type uM
                inst✝⁶ : CommRing R
                inst✝⁵ : AddCommGroup M
                inst✝⁴ : Module R M
                Q : QuadraticForm R M
                A : Type uA
                B : Type uB
                inst✝³ : Ring A
                inst✝² : Ring B
                inst✝¹ : Algebra R A
                inst✝ : Algebra R B
                f : CliffordAlgebra.EvenHom Q A
                c : R
                m : M
                a : Prod A (Subtype fun x => Membership.mem (CliffordAlgebra.even.lift.S f) x)
                m₃ : M
                ⊢ Eq (HMul.hMul ((f.bilin m₃) (HSMul.hSMul c m)) a.1) (HSMul.hSMul c (HMul.hMu …
              -/
              rw [LinearMap.map_smul, smul_mul_assoc]))
              /-
                🎉 no goals
              -/
    (fun _ _ _ => Prod.ext rfl (Subtype.ext <| LinearMap.ext fun _ => mul_add _ _ _))
    fun _ _ _ => Prod.ext rfl (Subtype.ext <| LinearMap.ext fun _ => mul_smul_comm _ _ _)


@[simp]
private theorem fst_fFold_fFold (m₁ m₂ : M) (x : A × S f) :
    (fFold f m₁ (fFold f m₂ x)).fst = f.bilin m₁ m₂ * x.fst :=
  rfl


@[simp]
private theorem snd_fFold_fFold (m₁ m₂ m₃ : M) (x : A × S f) :
    ((fFold f m₁ (fFold f m₂ x)).snd : M →ₗ[R] A) m₃ = f.bilin m₃ m₁ * (x.snd : M →ₗ[R] A) m₂ :=
  rfl


private theorem fFold_fFold (m : M) (x : A × S f) : fFold f m (fFold f m x) = Q m • x := by
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type uA
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    f : CliffordAlgebra.EvenHom Q A
    m : M
    x : Prod A (Subtype fun x => Membership.mem (CliffordAlgebra.even.lift.S f) x)
    ⊢ Eq (((CliffordAlgebra.even.lift.fFold f) m) (((CliffordAlgebra.even.lift.fFo …
  -/
  obtain ⟨a, ⟨g, hg⟩⟩ := x
  /-
    case mk.mk
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type uA
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    f : CliffordAlgebra.EvenHom Q A
    m : M
    a : A
    g : LinearMap (RingHom.id R) M A
    hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
    ⊢ Eq (((CliffordAlgebra.even.lift.fFold f) m) (((CliffordAlgebra.even.lift.fFo …
  -/
  ext : 2
    /-
      case mk.mk.fst
      R : Type uR
      M : Type uM
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      Q : QuadraticForm R M
      A : Type uA
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      f : CliffordAlgebra.EvenHom Q A
      m : M
      a : A
      g : LinearMap (RingHom.id R) M A
      hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
      ⊢ Eq (((CliffordAlgebra.even.lift.fFold f) m) (((CliffordAlgebra.even.lift.fFo …
    -/
  · change f.bilin m m * a = Q m • a
    /-
      case mk.mk.fst
      R : Type uR
      M : Type uM
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      Q : QuadraticForm R M
      A : Type uA
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      f : CliffordAlgebra.EvenHom Q A
      m : M
      a : A
      g : LinearMap (RingHom.id R) M A
      hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
      ⊢ Eq (HMul.hMul ((f.bilin m) m) a) (HSMul.hSMul (Q m) a)
    -/
    rw [Algebra.smul_def, f.contract]
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.snd.a
      R : Type uR
      M : Type uM
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      Q : QuadraticForm R M
      A : Type uA
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      f : CliffordAlgebra.EvenHom Q A
      m : M
      a : A
      g : LinearMap (RingHom.id R) M A
      hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
      ⊢ Eq ↑(((CliffordAlgebra.even.lift.fFold f) m) (((CliffordAlgebra.even.lift.fF …
    -/
  · ext m₁
    /-
      case mk.mk.snd.a.h
      R : Type uR
      M : Type uM
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      Q : QuadraticForm R M
      A : Type uA
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      f : CliffordAlgebra.EvenHom Q A
      m : M
      a : A
      g : LinearMap (RingHom.id R) M A
      hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
      m₁ : M
      ⊢ Eq (↑(((CliffordAlgebra.even.lift.fFold f) m) (((CliffordAlgebra.even.lift.f …
    -/
    change f.bilin _ _ * g m = Q m • g m₁
    /-
      case mk.mk.snd.a.h
      R : Type uR
      M : Type uM
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      Q : QuadraticForm R M
      A : Type uA
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      f : CliffordAlgebra.EvenHom Q A
      m : M
      a : A
      g : LinearMap (RingHom.id R) M A
      hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
      m₁ : M
      ⊢ Eq (HMul.hMul ((f.bilin m₁) m) (g m)) (HSMul.hSMul (Q m) (g m₁))
    -/
    refine Submodule.span_induction ?_ ?_ ?_ ?_ hg
      /-
        case mk.mk.snd.a.h.refine_1
        R : Type uR
        M : Type uM
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type uA
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : CliffordAlgebra.EvenHom Q A
        m : M
        a : A
        g : LinearMap (RingHom.id R) M A
        hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
        m₁ : M
        ⊢ ∀ (x : LinearMap (RingHom.id R) M A), Membership.mem (setOf fun f' => Exists …
      -/
    · rintro _ ⟨b, m₃, rfl⟩
      /-
        case mk.mk.snd.a.h.refine_1.intro.intro
        R : Type uR
        M : Type uM
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type uA
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : CliffordAlgebra.EvenHom Q A
        m : M
        a : A
        g : LinearMap (RingHom.id R) M A
        hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
        m₁ : M
        b : A
        m₃ : M
        ⊢ Eq (HMul.hMul ((f.bilin m₁) m) (((LinearMap.lcomp R A (f.bilin.flip m₃)) (Li …
      -/
      change f.bilin _ _ * (f.bilin _ _ * b) = Q m • (f.bilin _ _ * b)
      /-
        case mk.mk.snd.a.h.refine_1.intro.intro
        R : Type uR
        M : Type uM
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type uA
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : CliffordAlgebra.EvenHom Q A
        m : M
        a : A
        g : LinearMap (RingHom.id R) M A
        hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
        m₁ : M
        b : A
        m₃ : M
        ⊢ Eq (HMul.hMul ((f.bilin m₁) m) (HMul.hMul ((f.bilin m) m₃) b)) (HSMul.hSMul  …
      -/
      rw [← smul_mul_assoc, ← mul_assoc, f.contract_mid]
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.snd.a.h.refine_2
        R : Type uR
        M : Type uM
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type uA
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : CliffordAlgebra.EvenHom Q A
        m : M
        a : A
        g : LinearMap (RingHom.id R) M A
        hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
        m₁ : M
        ⊢ Eq (HMul.hMul ((f.bilin m₁) m) (0 m)) (HSMul.hSMul (Q m) (0 m₁))
      -/
    · change f.bilin m₁ m * 0 = Q m • (0 : A)  -- Porting note: `•` now needs the type of `0`
      /-
        case mk.mk.snd.a.h.refine_2
        R : Type uR
        M : Type uM
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type uA
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : CliffordAlgebra.EvenHom Q A
        m : M
        a : A
        g : LinearMap (RingHom.id R) M A
        hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
        m₁ : M
        ⊢ Eq (HMul.hMul ((f.bilin m₁) m) 0) (HSMul.hSMul (Q m) 0)
      -/
      rw [mul_zero, smul_zero]
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.snd.a.h.refine_3
        R : Type uR
        M : Type uM
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type uA
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : CliffordAlgebra.EvenHom Q A
        m : M
        a : A
        g : LinearMap (RingHom.id R) M A
        hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
        m₁ : M
        ⊢ ∀ (x y : LinearMap (RingHom.id R) M A), Membership.mem (Submodule.span R (se …
      -/
    · rintro x y _hx _hy ihx ihy
      /-
        case mk.mk.snd.a.h.refine_3
        R : Type uR
        M : Type uM
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type uA
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : CliffordAlgebra.EvenHom Q A
        m : M
        a : A
        g : LinearMap (RingHom.id R) M A
        hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
        m₁ : M
        x y : LinearMap (RingHom.id R) M A
        _hx : Membership.mem (Submodule.span R (setOf fun f' => Exists fun x => Exists …
        _hy : Membership.mem (Submodule.span R (setOf fun f' => Exists fun x => Exists …
        ihx : Eq (HMul.hMul ((f.bilin m₁) m) (x m)) (HSMul.hSMul (Q m) (x m₁))
        ihy : Eq (HMul.hMul ((f.bilin m₁) m) (y m)) (HSMul.hSMul (Q m) (y m₁))
        ⊢ Eq (HMul.hMul ((f.bilin m₁) m) ((HAdd.hAdd x y) m)) (HSMul.hSMul (Q m) ((HAd …
      -/
      rw [LinearMap.add_apply, LinearMap.add_apply, mul_add, smul_add, ihx, ihy]
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.snd.a.h.refine_4
        R : Type uR
        M : Type uM
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type uA
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : CliffordAlgebra.EvenHom Q A
        m : M
        a : A
        g : LinearMap (RingHom.id R) M A
        hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
        m₁ : M
        ⊢ ∀ (a : R) (x : LinearMap (RingHom.id R) M A), Membership.mem (Submodule.span …
      -/
    · rintro x hx _c ihx
      /-
        case mk.mk.snd.a.h.refine_4
        R : Type uR
        M : Type uM
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type uA
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : CliffordAlgebra.EvenHom Q A
        m : M
        a : A
        g : LinearMap (RingHom.id R) M A
        hg : Membership.mem (CliffordAlgebra.even.lift.S f) g
        m₁ : M
        x : R
        hx : LinearMap (RingHom.id R) M A
        _c : Membership.mem (Submodule.span R (setOf fun f' => Exists fun x => Exists  …
        ihx : Eq (HMul.hMul ((f.bilin m₁) m) (hx m)) (HSMul.hSMul (Q m) (hx m₁))
        ⊢ Eq (HMul.hMul ((f.bilin m₁) m) ((HSMul.hSMul x hx) m)) (HSMul.hSMul (Q m) (( …
      -/
      rw [LinearMap.smul_apply, LinearMap.smul_apply, mul_smul_comm, ihx, smul_comm]
      /-
        🎉 no goals
      -/

-- Porting note: In Lean 3, `aux_apply` isn't a simp lemma. I changed `{ attrs := [] }` to
-- `.lemmasOnly`, so that `aux_apply` isn't a simp lemma.

/-- The final auxiliary construction for `CliffordAlgebra.even.lift`. This map is the forwards
direction of that equivalence, but not in the fully-bundled form. -/
@[simps! (config := .lemmasOnly) apply]
def aux (f : EvenHom Q A) : CliffordAlgebra.even Q →ₗ[R] A := by
  /-
    R : Type uR
    M : Type uM
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    Q : QuadraticForm R M
    A : Type uA
    B : Type uB
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f✝ f : CliffordAlgebra.EvenHom Q A
    ⊢ LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (CliffordAlgebra.e …
  -/
  refine ?_ ∘ₗ (even Q).val.toLinearMap
  -- Porting note: added, can't be found otherwise
  /-
    R : Type uR
    M : Type uM
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    Q : QuadraticForm R M
    A : Type uA
    B : Type uB
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f✝ f : CliffordAlgebra.EvenHom Q A
    ⊢ LinearMap (RingHom.id R) (CliffordAlgebra Q) A
  -/
  letI : AddCommGroup (S f) := AddSubgroupClass.toAddCommGroup _
  /-
    R : Type uR
    M : Type uM
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    Q : QuadraticForm R M
    A : Type uA
    B : Type uB
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f✝ f : CliffordAlgebra.EvenHom Q A
    this : AddCommGroup (Subtype fun x => Membership.mem (CliffordAlgebra.even.lif …
    ⊢ LinearMap (RingHom.id R) (CliffordAlgebra Q) A
  -/
  exact LinearMap.fst R _ _ ∘ₗ foldr Q (fFold f) (fFold_fFold f) (1, 0)
  /-
    🎉 no goals
  -/


@[simp, nolint simpNF] -- Added `nolint simpNF` to avoid a timeout https://github.com/leanprover-community/mathlib4/pull/8386
theorem aux_one : aux f 1 = 1 :=
  congr_arg Prod.fst (foldr_one _ _ _ _)


@[simp, nolint simpNF] -- Added `nolint simpNF` to avoid a timeout https://github.com/leanprover-community/mathlib4/pull/8386
theorem aux_ι (m₁ m₂ : M) : aux f ((even.ι Q).bilin m₁ m₂) = f.bilin m₁ m₂ :=
  (congr_arg Prod.fst (foldr_mul _ _ _ _ _ _)).trans
    (by
      /-
        R : Type uR
        M : Type uM
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type uA
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : CliffordAlgebra.EvenHom Q A
        m₁ m₂ : M
        ⊢ Eq (((CliffordAlgebra.foldr Q (CliffordAlgebra.even.lift.fFold f) ⋯) (((Clif …
      -/
      rw [foldr_ι, foldr_ι]
      /-
        R : Type uR
        M : Type uM
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        Q : QuadraticForm R M
        A : Type uA
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        f : CliffordAlgebra.EvenHom Q A
        m₁ m₂ : M
        ⊢ Eq (((CliffordAlgebra.even.lift.fFold f) m₁) (((CliffordAlgebra.even.lift.fF …
      -/
      exact mul_one _)
      /-
        🎉 no goals
      -/


@[simp, nolint simpNF] -- Added `nolint simpNF` to avoid a timeout https://github.com/leanprover-community/mathlib4/pull/8386
theorem aux_algebraMap (r) (hr) : aux f ⟨algebraMap R _ r, hr⟩ = algebraMap R _ r :=
  (congr_arg Prod.fst (foldr_algebraMap _ _ _ _ _)).trans (Algebra.algebraMap_eq_smul_one r).symm


@[simp, nolint simpNF] -- Added `nolint simpNF` to avoid a timeout https://github.com/leanprover-community/mathlib4/pull/8386
theorem aux_mul (x y : even Q) : aux f (x * y) = aux f x * aux f y := by
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type uA
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    f : CliffordAlgebra.EvenHom Q A
    x y : Subtype fun x => Membership.mem (CliffordAlgebra.even Q) x
    ⊢ Eq ((CliffordAlgebra.even.lift.aux f) (HMul.hMul x y)) (HMul.hMul ((Clifford …
  -/
  cases' x with x x_property
  /-
    case mk
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type uA
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    f : CliffordAlgebra.EvenHom Q A
    y : Subtype fun x => Membership.mem (CliffordAlgebra.even Q) x
    x : CliffordAlgebra Q
    x_property : Membership.mem (CliffordAlgebra.even Q) x
    ⊢ Eq ((CliffordAlgebra.even.lift.aux f) (HMul.hMul ⟨x, x_property⟩ y)) (HMul.h …
  -/
  cases y
  /-
    case mk.mk
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type uA
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    f : CliffordAlgebra.EvenHom Q A
    x : CliffordAlgebra Q
    x_property : Membership.mem (CliffordAlgebra.even Q) x
    val✝ : CliffordAlgebra Q
    property✝ : Membership.mem (CliffordAlgebra.even Q) val✝
    ⊢ Eq ((CliffordAlgebra.even.lift.aux f) (HMul.hMul ⟨x, x_property⟩ ⟨val✝, prop …
  -/
  refine (congr_arg Prod.fst (foldr_mul _ _ _ _ _ _)).trans ?_
  /-
    case mk.mk
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : QuadraticForm R M
    A : Type uA
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    f : CliffordAlgebra.EvenHom Q A
    x : CliffordAlgebra Q
    x_property : Membership.mem (CliffordAlgebra.even Q) x
    val✝ : CliffordAlgebra Q
    property✝ : Membership.mem (CliffordAlgebra.even Q) val✝
    ⊢ Eq (((CliffordAlgebra.foldr Q (CliffordAlgebra.even.lift.fFold f) ⋯) (((Clif …
  -/
  dsimp only
  induction x, x_property using even_induction Q with
  | algebraMap r =>
    rw [foldr_algebraMap, aux_algebraMap]
    exact Algebra.smul_def r _
  | add x y hx hy ihx ihy =>
    rw [LinearMap.map_add, Prod.fst_add, ihx, ihy, ← add_mul, ← LinearMap.map_add]
    rfl
  | ι_mul_ι_mul m₁ m₂ x hx ih =>
    rw [aux_apply, foldr_mul, foldr_mul, foldr_ι, foldr_ι, fst_fFold_fFold, ih, ← mul_assoc,
      Subtype.coe_mk, foldr_mul, foldr_mul, foldr_ι, foldr_ι, fst_fFold_fFold]
    rfl


/-- Every algebra morphism from the even subalgebra is in one-to-one correspondence with a
bilinear map that sends duplicate arguments to the quadratic form, and contracts across
multiplication. -/
@[simps! symm_apply_bilin]
def even.lift : EvenHom Q A ≃ (CliffordAlgebra.even Q →ₐ[R] A) where
  toFun f := AlgHom.ofLinearMap (aux f) (aux_one f) (aux_mul f)
  invFun F := (even.ι Q).compr₂ F
  left_inv f := EvenHom.ext <| LinearMap.ext₂ <| even.lift.aux_ι f
  right_inv _ := even.algHom_ext Q <| EvenHom.ext <| LinearMap.ext₂ <| even.lift.aux_ι _

-- @[simp] -- Porting note: simpNF linter times out on this one

theorem even.lift_ι (f : EvenHom Q A) (m₁ m₂ : M) :
    even.lift Q f ((even.ι Q).bilin m₁ m₂) = f.bilin m₁ m₂ :=
  even.lift.aux_ι _ _ _


