/--
Given a surjective algebra homomorphism `f : P →ₐ[R] S` with square-zero kernel `I`,
and a section `g : S →ₐ[R] P` (as an algebra homomorphism),
we get an `R`-derivation `P → I` via `x ↦ x - g (f x)`.
-/
@[simps]
def derivationOfSectionOfKerSqZero (f : P →ₐ[R] S) (hf' : (RingHom.ker f) ^ 2 = ⊥) (g : S →ₐ[R] P)
    (hg : f.comp g = AlgHom.id R S) : Derivation R P (RingHom.ker f) where
  toFun x := ⟨x - g (f x), by
    /-
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      g✝ : AlgHom R S P
      f : AlgHom R P S
      hf' : Eq (HPow.hPow (RingHom.ker f) 2) Bot.bot
      g : AlgHom R S P
      hg : Eq (f.comp g) (AlgHom.id R S)
      x : P
      ⊢ Membership.mem (RingHom.ker f) (HSub.hSub x (g (f x)))
    -/
    simpa [RingHom.mem_ker, sub_eq_zero] using AlgHom.congr_fun hg.symm (f x)⟩
    /-
      🎉 no goals
    -/
                     /-
                       R P S : Type u
                       inst✝⁶ : CommRing R
                       inst✝⁵ : CommRing P
                       inst✝⁴ : CommRing S
                       inst✝³ : Algebra R P
                       inst✝² : Algebra P S
                       inst✝¹ : Algebra R S
                       inst✝ : IsScalarTower R P S
                       g✝ : AlgHom R S P
                       f : AlgHom R P S
                       hf' : Eq (HPow.hPow (RingHom.ker f) 2) Bot.bot
                       g : AlgHom R S P
                       hg : Eq (f.comp g) (AlgHom.id R S)
                       x y : P
                       ⊢ Eq ((fun x => ⟨HSub.hSub x (g (f x)), ⋯⟩) (HAdd.hAdd x y)) (HAdd.hAdd ((fun  …
                     -/
  map_add' x y := by simp only [map_add, AddMemClass.mk_add_mk, Subtype.mk.injEq]; ring
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  map_smul' x y := by
    /-
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      g✝ : AlgHom R S P
      f : AlgHom R P S
      hf' : Eq (HPow.hPow (RingHom.ker f) 2) Bot.bot
      g : AlgHom R S P
      hg : Eq (f.comp g) (AlgHom.id R S)
      x : R
      y : P
      ⊢ Eq ({ toFun := fun x => ⟨HSub.hSub x (g (f x)), ⋯⟩, map_add' := ⋯ }.toFun (H …
    -/
    ext
    simp only [Algebra.smul_def, map_mul, ← IsScalarTower.algebraMap_apply, AlgHom.commutes,
      RingHom.id_apply, Submodule.coe_smul_of_tower]
    /-
      case a
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      g✝ : AlgHom R S P
      f : AlgHom R P S
      hf' : Eq (HPow.hPow (RingHom.ker f) 2) Bot.bot
      g : AlgHom R S P
      hg : Eq (f.comp g) (AlgHom.id R S)
      x : R
      y : P
      ⊢ Eq (HSub.hSub (HMul.hMul ((algebraMap R P) x) y) (HMul.hMul ((algebraMap R P …
    -/
    ring
    /-
      🎉 no goals
    -/
  map_one_eq_zero' := by simp only [LinearMap.coe_mk, AddHom.coe_mk, map_one, sub_self,
    Submodule.mk_eq_zero]
  leibniz' a b := by
    have : (a - g (f a)) * (b - g (f b)) = 0 := by
      rw [← Ideal.mem_bot, ← hf', pow_two]
      apply Ideal.mul_mem_mul
      · simpa [RingHom.mem_ker, sub_eq_zero] using AlgHom.congr_fun hg.symm (f a)
      · simpa [RingHom.mem_ker, sub_eq_zero] using AlgHom.congr_fun hg.symm (f b)
    /-
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      g✝ : AlgHom R S P
      f : AlgHom R P S
      hf' : Eq (HPow.hPow (RingHom.ker f) 2) Bot.bot
      g : AlgHom R S P
      hg : Eq (f.comp g) (AlgHom.id R S)
      a b : P
      this : Eq (HMul.hMul (HSub.hSub a (g (f a))) (HSub.hSub b (g (f b)))) 0
      ⊢ Eq ({ toFun := fun x => ⟨HSub.hSub x (g (f x)), ⋯⟩, map_add' := ⋯, map_smul' …
    -/
    ext
    /-
      case a
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      g✝ : AlgHom R S P
      f : AlgHom R P S
      hf' : Eq (HPow.hPow (RingHom.ker f) 2) Bot.bot
      g : AlgHom R S P
      hg : Eq (f.comp g) (AlgHom.id R S)
      a b : P
      this : Eq (HMul.hMul (HSub.hSub a (g (f a))) (HSub.hSub b (g (f b)))) 0
      ⊢ Eq ↑({ toFun := fun x => ⟨HSub.hSub x (g (f x)), ⋯⟩, map_add' := ⋯, map_smul …
    -/
    rw [← sub_eq_zero]
    /-
      case a
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      g✝ : AlgHom R S P
      f : AlgHom R P S
      hf' : Eq (HPow.hPow (RingHom.ker f) 2) Bot.bot
      g : AlgHom R S P
      hg : Eq (f.comp g) (AlgHom.id R S)
      a b : P
      this : Eq (HMul.hMul (HSub.hSub a (g (f a))) (HSub.hSub b (g (f b)))) 0
      ⊢ Eq (HSub.hSub ↑({ toFun := fun x => ⟨HSub.hSub x (g (f x)), ⋯⟩, map_add' :=  …
    -/
    conv_rhs => rw [← neg_zero, ← this]
    simp only [LinearMap.coe_mk, AddHom.coe_mk, map_mul, SetLike.mk_smul_mk, smul_eq_mul, mul_sub,
      AddMemClass.mk_add_mk, sub_mul, neg_sub]
    /-
      case a
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      g✝ : AlgHom R S P
      f : AlgHom R P S
      hf' : Eq (HPow.hPow (RingHom.ker f) 2) Bot.bot
      g : AlgHom R S P
      hg : Eq (f.comp g) (AlgHom.id R S)
      a b : P
      this : Eq (HMul.hMul (HSub.hSub a (g (f a))) (HSub.hSub b (g (f b)))) 0
      ⊢ Eq (HSub.hSub (HSub.hSub (HMul.hMul a b) (HMul.hMul (g (f a)) (g (f b)))) (H …
    -/
    ring
    /-
      🎉 no goals
    -/


lemma isScalarTower_of_section_of_ker_sqZero :
    letI := g.toRingHom.toAlgebra; IsScalarTower P S (RingHom.ker (algebraMap P S)) := by
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    ⊢ IsScalarTower P S (Subtype fun x => Membership.mem (RingHom.ker (algebraMap  …
  -/
  letI := g.toRingHom.toAlgebra
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    this : Algebra S P := g.toAlgebra
    ⊢ IsScalarTower P S (Subtype fun x => Membership.mem (RingHom.ker (algebraMap  …
  -/
  constructor
  /-
    case smul_assoc
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    this : Algebra S P := g.toAlgebra
    ⊢ ∀ (x : P) (y : S) (z : Subtype fun x => Membership.mem (RingHom.ker (algebra …
  -/
  intro p s m
  /-
    case smul_assoc
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    this : Algebra S P := g.toAlgebra
    p : P
    s : S
    m : Subtype fun x => Membership.mem (RingHom.ker (algebraMap P S)) x
    ⊢ Eq (HSMul.hSMul (HSMul.hSMul p s) m) (HSMul.hSMul p (HSMul.hSMul s m))
  -/
  ext
  /-
    case smul_assoc.a
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    this : Algebra S P := g.toAlgebra
    p : P
    s : S
    m : Subtype fun x => Membership.mem (RingHom.ker (algebraMap P S)) x
    ⊢ Eq ↑(HSMul.hSMul (HSMul.hSMul p s) m) ↑(HSMul.hSMul p (HSMul.hSMul s m))
  -/
  show g (p • s) * m = p * (g s * m)
  /-
    case smul_assoc.a
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    this : Algebra S P := g.toAlgebra
    p : P
    s : S
    m : Subtype fun x => Membership.mem (RingHom.ker (algebraMap P S)) x
    ⊢ Eq (HMul.hMul (g (HSMul.hSMul p s)) ↑m) (HMul.hMul p (HMul.hMul (g s) ↑m))
  -/
  simp only [Algebra.smul_def, map_mul, mul_assoc, mul_left_comm _ (g s)]
  /-
    case smul_assoc.a
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    this : Algebra S P := g.toAlgebra
    p : P
    s : S
    m : Subtype fun x => Membership.mem (RingHom.ker (algebraMap P S)) x
    ⊢ Eq (HMul.hMul (g s) (HMul.hMul (g ((algebraMap P S) p)) ↑m)) (HMul.hMul (g s …
  -/
  congr 1
  /-
    case smul_assoc.a.e_a
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    this : Algebra S P := g.toAlgebra
    p : P
    s : S
    m : Subtype fun x => Membership.mem (RingHom.ker (algebraMap P S)) x
    ⊢ Eq (HMul.hMul (g ((algebraMap P S) p)) ↑m) (HMul.hMul p ↑m)
  -/
  rw [← sub_eq_zero, ← Ideal.mem_bot, ← hf', pow_two, ← sub_mul]
  /-
    case smul_assoc.a.e_a
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    this : Algebra S P := g.toAlgebra
    p : P
    s : S
    m : Subtype fun x => Membership.mem (RingHom.ker (algebraMap P S)) x
    ⊢ Membership.mem (HMul.hMul (RingHom.ker (algebraMap P S)) (RingHom.ker (algeb …
  -/
  refine Ideal.mul_mem_mul ?_ m.2
  /-
    case smul_assoc.a.e_a
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    this : Algebra S P := g.toAlgebra
    p : P
    s : S
    m : Subtype fun x => Membership.mem (RingHom.ker (algebraMap P S)) x
    ⊢ Membership.mem (RingHom.ker (algebraMap P S)) (HSub.hSub (g ((algebraMap P S …
  -/
  simpa [RingHom.mem_ker, sub_eq_zero] using AlgHom.congr_fun hg (algebraMap P S p)
  /-
    🎉 no goals
  -/


/--
Given a surjective algebra hom `f : P →ₐ[R] S` with square-zero kernel `I`,
and a section `g : S →ₐ[R] P` (as algebra homs),
we get a retraction of the injection `I → S ⊗[P] Ω[P/R]`.
-/
noncomputable
def retractionOfSectionOfKerSqZero : S ⊗[P] Ω[P⁄R] →ₗ[P] RingHom.ker (algebraMap P S) :=
  letI := g.toRingHom.toAlgebra
  haveI := isScalarTower_of_section_of_ker_sqZero g hf' hg
  letI f : _ →ₗ[P] RingHom.ker (algebraMap P S) := (derivationOfSectionOfKerSqZero
    (IsScalarTower.toAlgHom R P S) hf' g hg).liftKaehlerDifferential
  (f.liftBaseChange S).restrictScalars P


@[simp]
lemma retractionOfSectionOfKerSqZero_tmul_D (s : S) (t : P) :
    retractionOfSectionOfKerSqZero g hf' hg (s ⊗ₜ .D _ _ t) =
      g s * t - g s * g (algebraMap _ _ t) := by
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    s : S
    t : P
    ⊢ Eq (↑((retractionOfSectionOfKerSqZero g hf' hg) (TensorProduct.tmul P s ((Ka …
  -/
  letI := g.toRingHom.toAlgebra
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    s : S
    t : P
    this : Algebra S P := g.toAlgebra
    ⊢ Eq (↑((retractionOfSectionOfKerSqZero g hf' hg) (TensorProduct.tmul P s ((Ka …
  -/
  haveI := isScalarTower_of_section_of_ker_sqZero g hf' hg
  simp only [retractionOfSectionOfKerSqZero, AlgHom.toRingHom_eq_coe, LinearMap.coe_restrictScalars,
    LinearMap.liftBaseChange_tmul, SetLike.val_smul_of_tower]
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    s : S
    t : P
    this✝ : Algebra S P := g.toAlgebra
    this : IsScalarTower P S (Subtype fun x => Membership.mem (RingHom.ker (algebr …
    ⊢ Eq (HSMul.hSMul s ↑((derivationOfSectionOfKerSqZero (IsScalarTower.toAlgHom  …
  -/
  erw [Derivation.liftKaehlerDifferential_comp_D]
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    s : S
    t : P
    this✝ : Algebra S P := g.toAlgebra
    this : IsScalarTower P S (Subtype fun x => Membership.mem (RingHom.ker (algebr …
    ⊢ Eq (HSMul.hSMul s ↑((derivationOfSectionOfKerSqZero (IsScalarTower.toAlgHom  …
  -/
  exact mul_sub (g s) t (g (algebraMap P S t))
  /-
    🎉 no goals
  -/


lemma retractionOfSectionOfKerSqZero_comp_kerToTensor :
    (retractionOfSectionOfKerSqZero g hf' hg).comp (kerToTensor R P S) = LinearMap.id := by
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    g : AlgHom R S P
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hg : Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
    ⊢ Eq ((retractionOfSectionOfKerSqZero g hf' hg).comp (KaehlerDifferential.kerT …
  -/
  ext x; simp [RingHom.mem_ker.mp x.2]
         /-
           🎉 no goals
         -/


lemma sectionOfRetractionKerToTensorAux_prop (x y) (h : algebraMap P S x = algebraMap P S y) :
    x - l (1 ⊗ₜ .D _ _ x) = y - l (1 ⊗ₜ .D _ _ y) := by
  rw [sub_eq_iff_eq_add, sub_add_comm, ← sub_eq_iff_eq_add, ← Submodule.coe_sub,
    ← map_sub, ← tmul_sub, ← map_sub]
  /-
    R P S : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing P
    inst✝² : CommRing S
    inst✝¹ : Algebra R P
    inst✝ : Algebra P S
    l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Su …
    hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P S)) LinearMap.id
    x y : P
    h : Eq ((algebraMap P S) x) ((algebraMap P S) y)
    ⊢ Eq (HSub.hSub x y) ↑(l (TensorProduct.tmul P 1 ((KaehlerDifferential.D R P)  …
  -/
  exact congr_arg Subtype.val (LinearMap.congr_fun hl.symm ⟨x - y, by simp [RingHom.mem_ker, h]⟩)
  /-
    🎉 no goals
  -/


/--
Given a surjective algebra homomorphism `f : P →ₐ[R] S` with square-zero kernel `I`.
Let `σ` be an arbitrary (set-theoretic) section of `f`.
Suppose we have a retraction `l` of the injection `I →ₗ[P] S ⊗[P] Ω[P/R]`, then
`x ↦ σ x - l (1 ⊗ D (σ x))` is an algebra homomorphism and a section to `f`.
-/
noncomputable
def sectionOfRetractionKerToTensorAux : S →ₐ[R] P where
  toFun x := σ x - l (1 ⊗ₜ .D _ _ (σ x))
                 /-
                   R P S : Type u
                   inst✝⁶ : CommRing R
                   inst✝⁵ : CommRing P
                   inst✝⁴ : CommRing S
                   inst✝³ : Algebra R P
                   inst✝² : Algebra P S
                   l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Su …
                   hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P S)) LinearMap.id
                   σ : S → P
                   hσ : ∀ (x : S), Eq ((algebraMap P S) (σ x)) x
                   inst✝¹ : Algebra R S
                   inst✝ : IsScalarTower R P S
                   hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
                   ⊢ Eq ((fun x => HSub.hSub (σ x) ↑(l (TensorProduct.tmul P 1 ((KaehlerDifferent …
                 -/
  map_one' := by simp [sectionOfRetractionKerToTensorAux_prop l hl (σ 1) 1 (by simp [hσ])]
                 /-
                   🎉 no goals
                 -/
  map_mul' a b := by
    have (x y) : (l x).1 * (l y).1 = 0 := by
      rw [← Ideal.mem_bot, ← hf', pow_two]; exact Ideal.mul_mem_mul (l x).2 (l y).2
    simp only [sectionOfRetractionKerToTensorAux_prop l hl (σ (a * b)) (σ a * σ b) (by simp [hσ]),
      Derivation.leibniz, tmul_add, tmul_smul, map_add, map_smul, Submodule.coe_add,
      SetLike.val_smul, smul_eq_mul, mul_sub, sub_mul, this, sub_zero]
    /-
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Su …
      hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P S)) LinearMap.id
      σ : S → P
      hσ : ∀ (x : S), Eq ((algebraMap P S) (σ x)) x
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      a b : S
      this : ∀ (x y : TensorProduct P S (KaehlerDifferential R P)), Eq (HMul.hMul ↑( …
      ⊢ Eq (HSub.hSub (HMul.hMul (σ a) (σ b)) (HAdd.hAdd (HMul.hMul (σ a) ↑(l (Tenso …
    -/
    ring
    /-
      🎉 no goals
    -/
  map_add' a b := by
    simp only [sectionOfRetractionKerToTensorAux_prop l hl (σ (a + b)) (σ a + σ b) (by simp [hσ]),
      map_add, tmul_add, Submodule.coe_add, add_sub_add_comm]
                  /-
                    R P S : Type u
                    inst✝⁶ : CommRing R
                    inst✝⁵ : CommRing P
                    inst✝⁴ : CommRing S
                    inst✝³ : Algebra R P
                    inst✝² : Algebra P S
                    l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Su …
                    hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P S)) LinearMap.id
                    σ : S → P
                    hσ : ∀ (x : S), Eq ((algebraMap P S) (σ x)) x
                    inst✝¹ : Algebra R S
                    inst✝ : IsScalarTower R P S
                    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
                    ⊢ Eq ((↑{ toFun := fun x => HSub.hSub (σ x) ↑(l (TensorProduct.tmul P 1 ((Kaeh …
                  -/
  map_zero' := by simp [sectionOfRetractionKerToTensorAux_prop l hl (σ 0) 0 (by simp [hσ])]
                  /-
                    🎉 no goals
                  -/
  commutes' r := by
    simp [sectionOfRetractionKerToTensorAux_prop l hl
      (σ (algebraMap R S r)) (algebraMap R P r) (by simp [hσ, ← IsScalarTower.algebraMap_apply])]


lemma sectionOfRetractionKerToTensorAux_algebraMap (x : P) :
    sectionOfRetractionKerToTensorAux l hl σ hσ hf' (algebraMap P S x) = x - l (1 ⊗ₜ .D _ _ x) :=
                                                      /-
                                                        R P S : Type u
                                                        inst✝⁶ : CommRing R
                                                        inst✝⁵ : CommRing P
                                                        inst✝⁴ : CommRing S
                                                        inst✝³ : Algebra R P
                                                        inst✝² : Algebra P S
                                                        l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Su …
                                                        hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P S)) LinearMap.id
                                                        σ : S → P
                                                        hσ : ∀ (x : S), Eq ((algebraMap P S) (σ x)) x
                                                        inst✝¹ : Algebra R S
                                                        inst✝ : IsScalarTower R P S
                                                        hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
                                                        x : P
                                                        ⊢ Eq ((algebraMap P S) (σ ((algebraMap P S) x))) ((algebraMap P S) x)
                                                      -/
  sectionOfRetractionKerToTensorAux_prop l hl _ x (by simp [hσ])
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma toAlgHom_comp_sectionOfRetractionKerToTensorAux :
    (IsScalarTower.toAlgHom R P S).comp
      (sectionOfRetractionKerToTensorAux l hl σ hσ hf') = AlgHom.id _ _ := by
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Su …
    hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P S)) LinearMap.id
    σ : S → P
    hσ : ∀ (x : S), Eq ((algebraMap P S) (σ x)) x
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hf : Function.Surjective ⇑(algebraMap P S)
    ⊢ Eq ((IsScalarTower.toAlgHom R P S).comp (sectionOfRetractionKerToTensorAux l …
  -/
  ext x
  /-
    case H
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Su …
    hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P S)) LinearMap.id
    σ : S → P
    hσ : ∀ (x : S), Eq ((algebraMap P S) (σ x)) x
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hf : Function.Surjective ⇑(algebraMap P S)
    x : S
    ⊢ Eq (((IsScalarTower.toAlgHom R P S).comp (sectionOfRetractionKerToTensorAux  …
  -/
  obtain ⟨x, rfl⟩ := hf x
  /-
    case H.intro
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Su …
    hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P S)) LinearMap.id
    σ : S → P
    hσ : ∀ (x : S), Eq ((algebraMap P S) (σ x)) x
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hf : Function.Surjective ⇑(algebraMap P S)
    x : P
    ⊢ Eq (((IsScalarTower.toAlgHom R P S).comp (sectionOfRetractionKerToTensorAux  …
  -/
  simp [sectionOfRetractionKerToTensorAux_algebraMap, RingHom.mem_ker.mp]
  /-
    🎉 no goals
  -/


/--
Given a surjective algebra homomorphism `f : P →ₐ[R] S` with square-zero kernel `I`.
Suppose we have a retraction `l` of the injection `I →ₗ[P] S ⊗[P] Ω[P/R]`, then
`x ↦ σ x - l (1 ⊗ D (σ x))` is an algebra homomorphism and a section to `f`,
where `σ` is an arbitrary (set-theoretic) section of `f`
-/
noncomputable def sectionOfRetractionKerToTensor : S →ₐ[R] P :=
  sectionOfRetractionKerToTensorAux l hl _ (fun x ↦ (hf x).choose_spec) hf'


@[simp]
lemma sectionOfRetractionKerToTensor_algebraMap (x : P) :
    sectionOfRetractionKerToTensor l hl hf' hf (algebraMap P S x) = x - l (1 ⊗ₜ .D _ _ x) :=
  sectionOfRetractionKerToTensorAux_algebraMap l hl _ _ hf' x


@[simp]
lemma toAlgHom_comp_sectionOfRetractionKerToTensor :
    (IsScalarTower.toAlgHom R P S).comp
      (sectionOfRetractionKerToTensor l hl hf' hf) = AlgHom.id _ _ :=
  toAlgHom_comp_sectionOfRetractionKerToTensorAux (hf := hf) ..


/--
Given a surjective algebra homomorphism `f : P →ₐ[R] S` with square-zero kernel `I`,
there is a one-to-one correspondence between `P`-linear retractions of `I →ₗ[P] S ⊗[P] Ω[P/R]`
and algebra homomorphism sections of `f`.
-/
noncomputable
def retractionKerToTensorEquivSection :
    { l // l ∘ₗ (kerToTensor R P S) = LinearMap.id } ≃
      { g // (IsScalarTower.toAlgHom R P S).comp g = AlgHom.id R S } where
  toFun l := ⟨_, toAlgHom_comp_sectionOfRetractionKerToTensor _ l.2 hf' hf⟩
  invFun g := ⟨_, retractionOfSectionOfKerSqZero_comp_kerToTensor _ hf' g.2⟩
  left_inv l := by
    /-
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      l : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerToTensor R P S)) Linea …
      ⊢ Eq ((fun g => ⟨retractionOfSectionOfKerSqZero (↑g) hf' ⋯, ⋯⟩) ((fun l => ⟨se …
    -/
    ext s p
    /-
      case a.a.h.hf.H.a
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      l : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerToTensor R P S)) Linea …
      s : S
      p : P
      ⊢ Eq ↑((((TensorProduct.AlgebraTensorModule.curry ↑((fun g => ⟨retractionOfSec …
    -/
    obtain ⟨s, rfl⟩ := hf s
    have (x y) : (l.1 x).1 * (l.1 y).1 = 0 := by
      rw [← Ideal.mem_bot, ← hf', pow_two]; exact Ideal.mul_mem_mul (l.1 x).2 (l.1 y).2
    simp only [AlgebraTensorModule.curry_apply,
      Derivation.coe_comp, LinearMap.coe_comp, LinearMap.coe_restrictScalars, Derivation.coeFn_coe,
      Function.comp_apply, curry_apply, retractionOfSectionOfKerSqZero_tmul_D,
      sectionOfRetractionKerToTensor_algebraMap, ← mul_sub, sub_sub_cancel]
    /-
      case a.a.h.hf.H.a.intro
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      l : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerToTensor R P S)) Linea …
      p s : P
      this : ∀ (x y : TensorProduct P S (KaehlerDifferential R P)), Eq (HMul.hMul ↑( …
      ⊢ Eq (HMul.hMul (HSub.hSub s ↑(↑l (TensorProduct.tmul P 1 ((KaehlerDifferentia …
    -/
    rw [sub_mul]
    simp only [this, Algebra.algebraMap_eq_smul_one, ← smul_tmul', LinearMapClass.map_smul,
      SetLike.val_smul, smul_eq_mul, sub_zero]
                    /-
                      R P S : Type u
                      inst✝⁶ : CommRing R
                      inst✝⁵ : CommRing P
                      inst✝⁴ : CommRing S
                      inst✝³ : Algebra R P
                      inst✝² : Algebra P S
                      inst✝¹ : Algebra R S
                      inst✝ : IsScalarTower R P S
                      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
                      hf : Function.Surjective ⇑(algebraMap P S)
                      g : Subtype fun g => Eq ((IsScalarTower.toAlgHom R P S).comp g) (AlgHom.id R S)
                      ⊢ Eq ((fun l => ⟨sectionOfRetractionKerToTensor ↑l ⋯ hf' hf, ⋯⟩) ((fun g => ⟨r …
                    -/
  right_inv g := by ext s; obtain ⟨s, rfl⟩ := hf s; simp
                                                    /-
                                                      🎉 no goals
                                                    -/


variable (R P S) in
/--
Given a tower of algebras `S/P/R`, with `I = ker(P → S)`,
this is the `R`-derivative `P/I² → S ⊗[P] Ω[P⁄R]` given by `[x] ↦ 1 ⊗ D x`.
-/
noncomputable
def derivationQuotKerSq :
    Derivation R (P ⧸ (RingHom.ker (algebraMap P S) ^ 2)) (S ⊗[P] Ω[P⁄R]) := by
  letI := Submodule.liftQ ((RingHom.ker (algebraMap P S) ^ 2).restrictScalars R)
    (((mk P S _ 1).restrictScalars R).comp (KaehlerDifferential.D R P).toLinearMap)
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hf : Function.Surjective ⇑(algebraMap P S)
    this : LE.le (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (algebraMap  …
    ⊢ Derivation R (HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S …
  -/
  refine ⟨this ?_, ?_, ?_⟩
    /-
      case refine_1
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      this : LE.le (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (algebraMap  …
      ⊢ LE.le (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (algebraMap P S)) …
    -/
  · rintro x hx
    /-
      case refine_1
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      this : LE.le (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (algebraMap  …
      x : P
      hx : Membership.mem (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (alge …
      ⊢ Membership.mem (LinearMap.ker ((↑R ((TensorProduct.mk P S (KaehlerDifferenti …
    -/
    simp only [Submodule.restrictScalars_mem, pow_two] at hx
    simp only [LinearMap.mem_ker, LinearMap.coe_comp, LinearMap.coe_restrictScalars,
      Derivation.coeFn_coe, Function.comp_apply, mk_apply]
    /-
      case refine_1
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      this : LE.le (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (algebraMap  …
      x : P
      hx : Membership.mem (HMul.hMul (RingHom.ker (algebraMap P S)) (RingHom.ker (al …
      ⊢ Eq (TensorProduct.tmul P 1 ((KaehlerDifferential.D R P) x)) 0
    -/
    refine Submodule.smul_induction_on hx ?_ ?_
      /-
        case refine_1.refine_1
        R P S : Type u
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing P
        inst✝⁴ : CommRing S
        inst✝³ : Algebra R P
        inst✝² : Algebra P S
        inst✝¹ : Algebra R S
        inst✝ : IsScalarTower R P S
        hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
        hf : Function.Surjective ⇑(algebraMap P S)
        this : LE.le (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (algebraMap  …
        x : P
        hx : Membership.mem (HMul.hMul (RingHom.ker (algebraMap P S)) (RingHom.ker (al …
        ⊢ ∀ (r : P), Membership.mem (RingHom.ker (algebraMap P S)) r → ∀ (n : P), Memb …
      -/
    · intro x hx y hy
      simp only [smul_eq_mul, Derivation.leibniz, tmul_add, ← smul_tmul, Algebra.smul_def,
        mul_one, RingHom.mem_ker.mp hx, RingHom.mem_ker.mp hy, zero_tmul, zero_add]
      /-
        case refine_1.refine_2
        R P S : Type u
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing P
        inst✝⁴ : CommRing S
        inst✝³ : Algebra R P
        inst✝² : Algebra P S
        inst✝¹ : Algebra R S
        inst✝ : IsScalarTower R P S
        hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
        hf : Function.Surjective ⇑(algebraMap P S)
        this : LE.le (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (algebraMap  …
        x : P
        hx : Membership.mem (HMul.hMul (RingHom.ker (algebraMap P S)) (RingHom.ker (al …
        ⊢ ∀ (x y : P), Eq (TensorProduct.tmul P 1 ((KaehlerDifferential.D R P) x)) 0 → …
      -/
    · intro x y hx hy; simp only [map_add, hx, hy, tmul_add, zero_add]
                       /-
                         🎉 no goals
                       -/
    /-
      case refine_2
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      this : LE.le (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (algebraMap  …
      ⊢ Eq ((this ⋯) 1) 0
    -/
  · show (1 : S) ⊗ₜ[P] KaehlerDifferential.D R P 1 = 0; simp
                                                        /-
                                                          🎉 no goals
                                                        -/
    /-
      case refine_3
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      this : LE.le (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (algebraMap  …
      ⊢ ∀ (a b : HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S)) 2) …
    -/
  · intro a b
    /-
      case refine_3
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      this : LE.le (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (algebraMap  …
      a b : HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S)) 2)
      ⊢ Eq ((this ⋯) (HMul.hMul a b)) (HAdd.hAdd (HSMul.hSMul a ((this ⋯) b)) (HSMul …
    -/
    obtain ⟨a, rfl⟩ := Submodule.Quotient.mk_surjective _ a
    /-
      case refine_3.intro
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      this : LE.le (Submodule.restrictScalars R (HPow.hPow (RingHom.ker (algebraMap  …
      b : HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S)) 2)
      a : P
      ⊢ Eq ((this ⋯) (HMul.hMul (Submodule.Quotient.mk a) b)) (HAdd.hAdd (HSMul.hSMu …
    -/
    obtain ⟨b, rfl⟩ := Submodule.Quotient.mk_surjective _ b
    show (1 : S) ⊗ₜ[P] KaehlerDifferential.D R P (a * b) =
      Ideal.Quotient.mk _ a • ((1 : S) ⊗ₜ[P] KaehlerDifferential.D R P b) +
      Ideal.Quotient.mk _ b • ((1 : S) ⊗ₜ[P] KaehlerDifferential.D R P a)
    simp only [← Ideal.Quotient.algebraMap_eq, IsScalarTower.algebraMap_smul,
      Derivation.leibniz, tmul_add, tmul_smul]


@[simp]
lemma derivationQuotKerSq_mk (x : P) :
    derivationQuotKerSq R P S x = 1 ⊗ₜ .D R P x := rfl


variable (R P S) in
/--
Given a tower of algebras `S/P/R`, with `I = ker(P → S)` and `Q := P/I²`,
there is an isomorphism of `S`-modules `S ⊗[Q] Ω[Q/R] ≃ S ⊗[P] Ω[P/R]`.
-/
noncomputable
def tensorKaehlerQuotKerSqEquiv :
    S ⊗[P ⧸ (RingHom.ker (algebraMap P S) ^ 2)] Ω[(P ⧸ (RingHom.ker (algebraMap P S) ^ 2))⁄R] ≃ₗ[S]
      S ⊗[P] Ω[P⁄R] :=
  letI f₁ := (derivationQuotKerSq R P S).liftKaehlerDifferential
  letI f₂ := AlgebraTensorModule.lift ((LinearMap.ringLmapEquivSelf S S _).symm f₁)
  letI f₃ := KaehlerDifferential.map R R P (P ⧸ (RingHom.ker (algebraMap P S) ^ 2))
  letI f₄ := ((mk (P ⧸ RingHom.ker (algebraMap P S) ^ 2) S _ 1).restrictScalars P).comp f₃
  letI f₅ := AlgebraTensorModule.lift ((LinearMap.ringLmapEquivSelf S S _).symm f₄)
  { __ := f₂
    invFun := f₅
    left_inv := by
      /-
        R P S : Type u
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing P
        inst✝⁴ : CommRing S
        inst✝³ : Algebra R P
        inst✝² : Algebra P S
        inst✝¹ : Algebra R S
        inst✝ : IsScalarTower R P S
        hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
        hf : Function.Surjective ⇑(algebraMap P S)
        f₁ : LinearMap (RingHom.id (HasQuotient.Quotient P (HPow.hPow (RingHom.ker (al …
        f₂ : LinearMap (RingHom.id S) (TensorProduct (HasQuotient.Quotient P (HPow.hPo …
        f₃ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (KaehlerDifferential R …
        f₄ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (TensorProduct (HasQuo …
        f₅ : LinearMap (RingHom.id S) (TensorProduct P S (KaehlerDifferential R P)) (T …
        ⊢ Function.LeftInverse (⇑f₅) __spread✝⁻⁰.toFun
      -/
      suffices f₅.comp f₂ = LinearMap.id from LinearMap.congr_fun this
      /-
        R P S : Type u
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing P
        inst✝⁴ : CommRing S
        inst✝³ : Algebra R P
        inst✝² : Algebra P S
        inst✝¹ : Algebra R S
        inst✝ : IsScalarTower R P S
        hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
        hf : Function.Surjective ⇑(algebraMap P S)
        f₁ : LinearMap (RingHom.id (HasQuotient.Quotient P (HPow.hPow (RingHom.ker (al …
        f₂ : LinearMap (RingHom.id S) (TensorProduct (HasQuotient.Quotient P (HPow.hPo …
        f₃ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (KaehlerDifferential R …
        f₄ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (TensorProduct (HasQuo …
        f₅ : LinearMap (RingHom.id S) (TensorProduct P S (KaehlerDifferential R P)) (T …
        ⊢ Eq (f₅.comp f₂) LinearMap.id
      -/
      ext a
      /-
        case a.h.hf.H
        R P S : Type u
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing P
        inst✝⁴ : CommRing S
        inst✝³ : Algebra R P
        inst✝² : Algebra P S
        inst✝¹ : Algebra R S
        inst✝ : IsScalarTower R P S
        hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
        hf : Function.Surjective ⇑(algebraMap P S)
        f₁ : LinearMap (RingHom.id (HasQuotient.Quotient P (HPow.hPow (RingHom.ker (al …
        f₂ : LinearMap (RingHom.id S) (TensorProduct (HasQuotient.Quotient P (HPow.hPo …
        f₃ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (KaehlerDifferential R …
        f₄ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (TensorProduct (HasQuo …
        f₅ : LinearMap (RingHom.id S) (TensorProduct P S (KaehlerDifferential R P)) (T …
        a : HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S)) 2)
        ⊢ Eq ((((TensorProduct.AlgebraTensorModule.curry (f₅.comp f₂)) 1).compDer (Kae …
      -/
      obtain ⟨a, rfl⟩ := Ideal.Quotient.mk_surjective a
      /-
        case a.h.hf.H.intro
        R P S : Type u
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing P
        inst✝⁴ : CommRing S
        inst✝³ : Algebra R P
        inst✝² : Algebra P S
        inst✝¹ : Algebra R S
        inst✝ : IsScalarTower R P S
        hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
        hf : Function.Surjective ⇑(algebraMap P S)
        f₁ : LinearMap (RingHom.id (HasQuotient.Quotient P (HPow.hPow (RingHom.ker (al …
        f₂ : LinearMap (RingHom.id S) (TensorProduct (HasQuotient.Quotient P (HPow.hPo …
        f₃ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (KaehlerDifferential R …
        f₄ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (TensorProduct (HasQuo …
        f₅ : LinearMap (RingHom.id S) (TensorProduct P S (KaehlerDifferential R P)) (T …
        a : P
        ⊢ Eq ((((TensorProduct.AlgebraTensorModule.curry (f₅.comp f₂)) 1).compDer (Kae …
      -/
      simp [f₁, f₂, f₃, f₄, f₅]
      /-
        🎉 no goals
      -/
    right_inv := by
      /-
        R P S : Type u
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing P
        inst✝⁴ : CommRing S
        inst✝³ : Algebra R P
        inst✝² : Algebra P S
        inst✝¹ : Algebra R S
        inst✝ : IsScalarTower R P S
        hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
        hf : Function.Surjective ⇑(algebraMap P S)
        f₁ : LinearMap (RingHom.id (HasQuotient.Quotient P (HPow.hPow (RingHom.ker (al …
        f₂ : LinearMap (RingHom.id S) (TensorProduct (HasQuotient.Quotient P (HPow.hPo …
        f₃ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (KaehlerDifferential R …
        f₄ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (TensorProduct (HasQuo …
        f₅ : LinearMap (RingHom.id S) (TensorProduct P S (KaehlerDifferential R P)) (T …
        ⊢ Function.RightInverse (⇑f₅) __spread✝⁻⁰.toFun
      -/
      suffices f₂.comp f₅ = LinearMap.id from LinearMap.congr_fun this
      /-
        R P S : Type u
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing P
        inst✝⁴ : CommRing S
        inst✝³ : Algebra R P
        inst✝² : Algebra P S
        inst✝¹ : Algebra R S
        inst✝ : IsScalarTower R P S
        hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
        hf : Function.Surjective ⇑(algebraMap P S)
        f₁ : LinearMap (RingHom.id (HasQuotient.Quotient P (HPow.hPow (RingHom.ker (al …
        f₂ : LinearMap (RingHom.id S) (TensorProduct (HasQuotient.Quotient P (HPow.hPo …
        f₃ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (KaehlerDifferential R …
        f₄ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (TensorProduct (HasQuo …
        f₅ : LinearMap (RingHom.id S) (TensorProduct P S (KaehlerDifferential R P)) (T …
        ⊢ Eq (f₂.comp f₅) LinearMap.id
      -/
      ext a
      /-
        case a.h.hf.H
        R P S : Type u
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing P
        inst✝⁴ : CommRing S
        inst✝³ : Algebra R P
        inst✝² : Algebra P S
        inst✝¹ : Algebra R S
        inst✝ : IsScalarTower R P S
        hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
        hf : Function.Surjective ⇑(algebraMap P S)
        f₁ : LinearMap (RingHom.id (HasQuotient.Quotient P (HPow.hPow (RingHom.ker (al …
        f₂ : LinearMap (RingHom.id S) (TensorProduct (HasQuotient.Quotient P (HPow.hPo …
        f₃ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (KaehlerDifferential R …
        f₄ : LinearMap (RingHom.id P) (KaehlerDifferential R P) (TensorProduct (HasQuo …
        f₅ : LinearMap (RingHom.id S) (TensorProduct P S (KaehlerDifferential R P)) (T …
        a : P
        ⊢ Eq ((((TensorProduct.AlgebraTensorModule.curry (f₂.comp f₅)) 1).compDer (Kae …
      -/
      simp [f₁, f₂, f₃, f₄, f₅] }
      /-
        🎉 no goals
      -/


@[simp]
lemma tensorKaehlerQuotKerSqEquiv_tmul_D (s t) :
    tensorKaehlerQuotKerSqEquiv R P S (s ⊗ₜ .D _ _ (Ideal.Quotient.mk _ t)) = s ⊗ₜ .D _ _ t := by
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    s : S
    t : P
    ⊢ Eq ((tensorKaehlerQuotKerSqEquiv R P S) (TensorProduct.tmul (HasQuotient.Quo …
  -/
  show s • (derivationQuotKerSq R P S).liftKaehlerDifferential (.D _ _ (Ideal.Quotient.mk _ t)) = _
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    s : S
    t : P
    ⊢ Eq (HSMul.hSMul s ((derivationQuotKerSq R P S).liftKaehlerDifferential ((Kae …
  -/
  simp [smul_tmul']
  /-
    🎉 no goals
  -/


@[simp]
lemma tensorKaehlerQuotKerSqEquiv_symm_tmul_D (s t) :
    (tensorKaehlerQuotKerSqEquiv R P S).symm (s ⊗ₜ .D _ _ t) =
      s ⊗ₜ .D _ _ (Ideal.Quotient.mk _ t) := by
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    s : S
    t : P
    ⊢ Eq ((tensorKaehlerQuotKerSqEquiv R P S).symm (TensorProduct.tmul P s ((Kaehl …
  -/
  apply (tensorKaehlerQuotKerSqEquiv R P S).injective
  /-
    case a
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    s : S
    t : P
    ⊢ Eq ((tensorKaehlerQuotKerSqEquiv R P S) ((tensorKaehlerQuotKerSqEquiv R P S) …
  -/
  simp
  /-
    🎉 no goals
  -/


/--
Given a surjective algebra homomorphism `f : P →ₐ[R] S` with kernel `I`,
there is a one-to-one correspondence between `P`-linear retractions of `I/I² →ₗ[P] S ⊗[P] Ω[P/R]`
and algebra homomorphism sections of `f‾ : P/I² → S`.
-/
noncomputable
def retractionKerCotangentToTensorEquivSection :
    { l // l ∘ₗ (kerCotangentToTensor R P S) = LinearMap.id } ≃
      { g // (IsScalarTower.toAlgHom R P S).kerSquareLift.comp g = AlgHom.id R S } := by
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hf : Function.Surjective ⇑(algebraMap P S)
    ⊢ Equiv (Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor …
  -/
  let P' := P ⧸ (RingHom.ker (algebraMap P S) ^ 2)
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hf : Function.Surjective ⇑(algebraMap P S)
    P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
    ⊢ Equiv (Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor …
  -/
  have h₁ : Surjective (algebraMap P' S) := Function.Surjective.of_comp (g := algebraMap P P') hf
  have h₂ : RingHom.ker (algebraMap P' S) ^ 2 = ⊥ := by
    rw [RingHom.algebraMap_toAlgebra, AlgHom.ker_kerSquareLift, Ideal.cotangentIdeal_square]
  let e₁ : (RingHom.ker (algebraMap P S)).Cotangent ≃ₗ[P] (RingHom.ker (algebraMap P' S)) :=
    (Ideal.cotangentEquivIdeal _).trans ((LinearEquiv.ofEq _ _
      (IsScalarTower.toAlgHom R P S).ker_kerSquareLift.symm).restrictScalars P)
  let e₂ : S ⊗[P'] Ω[P'⁄R] ≃ₗ[P] S ⊗[P] Ω[P⁄R] :=
    (tensorKaehlerQuotKerSqEquiv R P S).restrictScalars P
  have H : kerCotangentToTensor R P S =
      e₂.toLinearMap ∘ₗ (kerToTensor R P' S ).restrictScalars P ∘ₗ e₁.toLinearMap := by
    ext x
    obtain ⟨x, rfl⟩ := Ideal.toCotangent_surjective _ x
    exact (tensorKaehlerQuotKerSqEquiv_tmul_D 1 x.1).symm
  /-
    R P S : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing P
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R P
    inst✝² : Algebra P S
    inst✝¹ : Algebra R S
    inst✝ : IsScalarTower R P S
    hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
    hf : Function.Surjective ⇑(algebraMap P S)
    P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
    h₁ : Function.Surjective ⇑(algebraMap P' S)
    h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
    e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
    e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
    H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
    ⊢ Equiv (Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor …
  -/
  refine Equiv.trans ?_ (retractionKerToTensorEquivSection (R := R) h₂ h₁)
  refine ⟨fun ⟨l, hl⟩ ↦ ⟨⟨(e₁.toLinearMap ∘ₗ l ∘ₗ e₂.toLinearMap).toAddMonoidHom, ?_⟩, ?_⟩,
    fun ⟨l, hl⟩ ↦ ⟨e₁.symm.toLinearMap ∘ₗ l.restrictScalars P ∘ₗ e₂.symm.toLinearMap, ?_⟩, ?_, ?_⟩
    /-
      case refine_1
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P …
      l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Ri …
      hl : Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P S)) LinearMap.id
      ⊢ ∀ (m : P') (x : TensorProduct P' S (KaehlerDifferential R P')), Eq ((↑((↑e₁) …
    -/
  · rintro x y
    /-
      case refine_1
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P …
      l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Ri …
      hl : Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P S)) LinearMap.id
      x : P'
      y : TensorProduct P' S (KaehlerDifferential R P')
      ⊢ Eq ((↑((↑e₁).comp (l.comp ↑e₂)).toAddMonoidHom).toFun (HSMul.hSMul x y)) (HS …
    -/
    obtain ⟨x, rfl⟩ := Ideal.Quotient.mk_surjective x
    /-
      case refine_1.intro
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P …
      l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Ri …
      hl : Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P S)) LinearMap.id
      y : TensorProduct P' S (KaehlerDifferential R P')
      x : P
      ⊢ Eq ((↑((↑e₁).comp (l.comp ↑e₂)).toAddMonoidHom).toFun (HSMul.hSMul ((Ideal.Q …
    -/
    simp only [P', ← Ideal.Quotient.algebraMap_eq, IsScalarTower.algebraMap_smul]
    /-
      case refine_1.intro
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P …
      l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Ri …
      hl : Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P S)) LinearMap.id
      y : TensorProduct P' S (KaehlerDifferential R P')
      x : P
      ⊢ Eq ((↑((↑e₁).comp (l.comp ↑e₂)).toAddMonoidHom).toFun (HSMul.hSMul x y)) (HS …
    -/
    exact (e₁.toLinearMap ∘ₗ l ∘ₗ e₂.toLinearMap).map_smul x y
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P …
      l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Ri …
      hl : Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P S)) LinearMap.id
      ⊢ Eq ({ toAddHom := ↑((↑e₁).comp (l.comp ↑e₂)).toAddMonoidHom, map_smul' := ⋯  …
    -/
  · ext1 x
    /-
      case refine_2.h
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P …
      l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Ri …
      hl : Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P S)) LinearMap.id
      x : Subtype fun x => Membership.mem (RingHom.ker (algebraMap P' S)) x
      ⊢ Eq (({ toAddHom := ↑((↑e₁).comp (l.comp ↑e₂)).toAddMonoidHom, map_smul' := ⋯ …
    -/
    rw [H] at hl
    /-
      case refine_2.h
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P …
      l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Ri …
      hl : Eq (l.comp ((↑e₂).comp ((↑P (KaehlerDifferential.kerToTensor R P' S)).com …
      x : Subtype fun x => Membership.mem (RingHom.ker (algebraMap P' S)) x
      ⊢ Eq (({ toAddHom := ↑((↑e₁).comp (l.comp ↑e₂)).toAddMonoidHom, map_smul' := ⋯ …
    -/
    obtain ⟨x, rfl⟩ := e₁.surjective x
    /-
      case refine_2.h.intro
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P …
      l : LinearMap (RingHom.id P) (TensorProduct P S (KaehlerDifferential R P)) (Ri …
      hl : Eq (l.comp ((↑e₂).comp ((↑P (KaehlerDifferential.kerToTensor R P' S)).com …
      x : (RingHom.ker (algebraMap P S)).Cotangent
      ⊢ Eq (({ toAddHom := ↑((↑e₁).comp (l.comp ↑e₂)).toAddMonoidHom, map_smul' := ⋯ …
    -/
    exact DFunLike.congr_arg e₁ (LinearMap.congr_fun hl x)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerToTensor R P' S)) Lin …
      l : LinearMap (RingHom.id P') (TensorProduct P' S (KaehlerDifferential R P'))  …
      hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P' S)) LinearMap.id
      ⊢ Eq (((↑e₁.symm).comp ((↑P l).comp ↑e₂.symm)).comp (KaehlerDifferential.kerCo …
    -/
  · ext x
    /-
      case refine_3.h
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerToTensor R P' S)) Lin …
      l : LinearMap (RingHom.id P') (TensorProduct P' S (KaehlerDifferential R P'))  …
      hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P' S)) LinearMap.id
      x : (RingHom.ker (algebraMap P S)).Cotangent
      ⊢ Eq ((((↑e₁.symm).comp ((↑P l).comp ↑e₂.symm)).comp (KaehlerDifferential.kerC …
    -/
    rw [H]
    /-
      case refine_3.h
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerToTensor R P' S)) Lin …
      l : LinearMap (RingHom.id P') (TensorProduct P' S (KaehlerDifferential R P'))  …
      hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P' S)) LinearMap.id
      x : (RingHom.ker (algebraMap P S)).Cotangent
      ⊢ Eq ((((↑e₁.symm).comp ((↑P l).comp ↑e₂.symm)).comp ((↑e₂).comp ((↑P (Kaehler …
    -/
    apply e₁.injective
    simp only [LinearMap.coe_comp, LinearEquiv.coe_coe, LinearMap.coe_restrictScalars,
      Function.comp_apply, LinearEquiv.symm_apply_apply, LinearMap.id_coe, id_eq,
      LinearEquiv.apply_symm_apply]
    /-
      case refine_3.h.a
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      x✝ : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerToTensor R P' S)) Lin …
      l : LinearMap (RingHom.id P') (TensorProduct P' S (KaehlerDifferential R P'))  …
      hl : Eq (l.comp (KaehlerDifferential.kerToTensor R P' S)) LinearMap.id
      x : (RingHom.ker (algebraMap P S)).Cotangent
      ⊢ Eq (l ((KaehlerDifferential.kerToTensor R P' S) (e₁ x))) (e₁ x)
    -/
    exact LinearMap.congr_fun hl (e₁ x)
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      ⊢ Function.LeftInverse (fun x => retractionKerCotangentToTensorEquivSection.ma …
    -/
  · intro f
    /-
      case refine_4
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      f : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R P  …
      ⊢ Eq ((fun x => retractionKerCotangentToTensorEquivSection.match_2 (fun x => S …
    -/
    ext x
    simp only [AlgebraTensorModule.curry_apply, Derivation.coe_comp, LinearMap.coe_comp,
      LinearMap.coe_restrictScalars, Derivation.coeFn_coe, Function.comp_apply, curry_apply,
      LinearEquiv.coe_coe, LinearMap.coe_mk, AddHom.coe_coe, LinearMap.toAddMonoidHom_coe,
      LinearEquiv.apply_symm_apply, LinearEquiv.symm_apply_apply]
    /-
      case refine_5
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      ⊢ Function.RightInverse (fun x => retractionKerCotangentToTensorEquivSection.m …
    -/
  · intro f
    /-
      case refine_5
      R P S : Type u
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing P
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R P
      inst✝² : Algebra P S
      inst✝¹ : Algebra R S
      inst✝ : IsScalarTower R P S
      hf' : Eq (HPow.hPow (RingHom.ker (algebraMap P S)) 2) Bot.bot
      hf : Function.Surjective ⇑(algebraMap P S)
      P' : Type u := HasQuotient.Quotient P (HPow.hPow (RingHom.ker (algebraMap P S) …
      h₁ : Function.Surjective ⇑(algebraMap P' S)
      h₂ : Eq (HPow.hPow (RingHom.ker (algebraMap P' S)) 2) Bot.bot
      e₁ : LinearEquiv (RingHom.id P) (RingHom.ker (algebraMap P S)).Cotangent (Subt …
      e₂ : LinearEquiv (RingHom.id P) (TensorProduct P' S (KaehlerDifferential R P') …
      H : Eq (KaehlerDifferential.kerCotangentToTensor R P S) ((↑e₂).comp ((↑P (Kaeh …
      f : Subtype fun l => Eq (l.comp (KaehlerDifferential.kerToTensor R P' S)) Line …
      ⊢ Eq ((fun x => retractionKerCotangentToTensorEquivSection.match_1 (fun x => S …
    -/
    ext x
    simp only [AlgebraTensorModule.curry_apply, Derivation.coe_comp, LinearMap.coe_comp,
      LinearMap.coe_restrictScalars, Derivation.coeFn_coe, Function.comp_apply, curry_apply,
      LinearMap.coe_mk, AddHom.coe_coe, LinearMap.toAddMonoidHom_coe, LinearEquiv.coe_coe,
      LinearEquiv.symm_apply_apply, LinearEquiv.apply_symm_apply]


include hf in
/--
Given a formally smooth `R`-algebra `P` and a surjective algebra homomorphism `f : P →ₐ[R] S`
with kernel `I` (typically a presentation `R[X] → S`),
`S` is formally smooth iff the `P`-linear map `I/I² → S ⊗[P] Ω[P⁄R]` is split injective.
-/
@[stacks 031I]
theorem Algebra.FormallySmooth.iff_split_injection :
    Algebra.FormallySmooth R S ↔ ∃ l, l ∘ₗ (kerCotangentToTensor R P S) = LinearMap.id := by
  /-
    R P S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing P
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R P
    inst✝³ : Algebra P S
    inst✝² : Algebra R S
    inst✝¹ : IsScalarTower R P S
    hf : Function.Surjective ⇑(algebraMap P S)
    inst✝ : Algebra.FormallySmooth R P
    ⊢ Iff (Algebra.FormallySmooth R S) (Exists fun l => Eq (l.comp (KaehlerDiffere …
  -/
  have := (retractionKerCotangentToTensorEquivSection (R := R) hf).nonempty_congr
  /-
    R P S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing P
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R P
    inst✝³ : Algebra P S
    inst✝² : Algebra R S
    inst✝¹ : IsScalarTower R P S
    hf : Function.Surjective ⇑(algebraMap P S)
    inst✝ : Algebra.FormallySmooth R P
    this : Iff (Nonempty (Subtype fun l => Eq (l.comp (KaehlerDifferential.kerCota …
    ⊢ Iff (Algebra.FormallySmooth R S) (Exists fun l => Eq (l.comp (KaehlerDiffere …
  -/
  simp only [nonempty_subtype] at this
  /-
    R P S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing P
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R P
    inst✝³ : Algebra P S
    inst✝² : Algebra R S
    inst✝¹ : IsScalarTower R P S
    hf : Function.Surjective ⇑(algebraMap P S)
    inst✝ : Algebra.FormallySmooth R P
    this : Iff (Exists fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTens …
    ⊢ Iff (Algebra.FormallySmooth R S) (Exists fun l => Eq (l.comp (KaehlerDiffere …
  -/
  rw [this, ← Algebra.FormallySmooth.iff_split_surjection _ hf]
  /-
    🎉 no goals
  -/


include hf in
/--
Given a formally smooth `R`-algebra `P` and a surjective algebra homomorphism `f : P →ₐ[R] S`
with kernel `I` (typically a presentation `R[X] → S`),
then `S` is formally smooth iff `I/I² → S ⊗[P] Ω[S⁄R]` is injective and
`S ⊗[P] Ω[P⁄R] → Ω[S⁄R]` is split surjective.
-/
theorem Algebra.FormallySmooth.iff_injective_and_split :
    Algebra.FormallySmooth R S ↔ Function.Injective (kerCotangentToTensor R P S) ∧
      ∃ l, (KaehlerDifferential.mapBaseChange R P S) ∘ₗ l = LinearMap.id := by
  /-
    R P S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing P
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R P
    inst✝³ : Algebra P S
    inst✝² : Algebra R S
    inst✝¹ : IsScalarTower R P S
    hf : Function.Surjective ⇑(algebraMap P S)
    inst✝ : Algebra.FormallySmooth R P
    ⊢ Iff (Algebra.FormallySmooth R S) (And (Function.Injective ⇑(KaehlerDifferent …
  -/
  rw [Algebra.FormallySmooth.iff_split_injection hf]
  /-
    R P S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing P
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R P
    inst✝³ : Algebra P S
    inst✝² : Algebra R S
    inst✝¹ : IsScalarTower R P S
    hf : Function.Surjective ⇑(algebraMap P S)
    inst✝ : Algebra.FormallySmooth R P
    ⊢ Iff (Exists fun l => Eq (l.comp (KaehlerDifferential.kerCotangentToTensor R  …
  -/
  refine (and_iff_right (KaehlerDifferential.mapBaseChange_surjective R _ _ hf)).symm.trans ?_
  refine Iff.trans (((exact_kerCotangentToTensor_mapBaseChange R _ _ hf).split_tfae'
    (g := (KaehlerDifferential.mapBaseChange R P S).restrictScalars P)).out 1 0)
    (and_congr Iff.rfl ?_)
  /-
    R P S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing P
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R P
    inst✝³ : Algebra P S
    inst✝² : Algebra R S
    inst✝¹ : IsScalarTower R P S
    hf : Function.Surjective ⇑(algebraMap P S)
    inst✝ : Algebra.FormallySmooth R P
    ⊢ Iff (Exists fun l => Eq ((↑P (KaehlerDifferential.mapBaseChange R P S)).comp …
  -/
  rw [(LinearMap.extendScalarsOfSurjectiveEquiv hf).surjective.exists]
  simp only [LinearMap.ext_iff, LinearMap.coe_comp, LinearMap.coe_restrictScalars,
    Function.comp_apply, LinearMap.extendScalarsOfSurjective_apply, LinearMap.id_coe, id_eq]


private theorem Algebra.FormallySmooth.iff_injective_and_projective' :
    letI : Algebra (MvPolynomial S R) S := (MvPolynomial.aeval _root_.id).toAlgebra
    Algebra.FormallySmooth R S ↔
        Function.Injective (kerCotangentToTensor R (MvPolynomial S R) S) ∧
        Module.Projective S (Ω[S⁄R]) := by
  /-
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Iff (Algebra.FormallySmooth R S) (And (Function.Injective ⇑(KaehlerDifferent …
  -/
  letI : Algebra (MvPolynomial S R) S := (MvPolynomial.aeval _root_.id).toAlgebra
  have : Function.Surjective (algebraMap (MvPolynomial S R) S) :=
    fun x ↦ ⟨.X x, MvPolynomial.aeval_X _ _⟩
  rw [Algebra.FormallySmooth.iff_injective_and_split this,
    ← Module.Projective.iff_split_of_projective]
  /-
    case hs
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    this✝ : Algebra (MvPolynomial S R) S := (MvPolynomial.aeval _root_.id).toAlgebra
    this : Function.Surjective ⇑(algebraMap (MvPolynomial S R) S)
    ⊢ Function.Surjective ⇑(KaehlerDifferential.mapBaseChange R (MvPolynomial S R) …
  -/
  exact KaehlerDifferential.mapBaseChange_surjective _ _ _ this
  /-
    🎉 no goals
  -/


instance : Module.Projective P (Ω[P⁄R]) :=
  (Algebra.FormallySmooth.iff_injective_and_projective'.mp ‹_›).2


include hf in
/--
Given a formally smooth `R`-algebra `P` and a surjective algebra homomorphism `f : P →ₐ[R] S`
with kernel `I` (typically a presentation `R[X] → S`),
then `S` is formally smooth iff `I/I² → S ⊗[P] Ω[P⁄R]` is injective and `Ω[S/R]` is projective.
-/
theorem Algebra.FormallySmooth.iff_injective_and_projective :
    Algebra.FormallySmooth R S ↔
        Function.Injective (kerCotangentToTensor R P S) ∧ Module.Projective S (Ω[S⁄R]) := by
  rw [Algebra.FormallySmooth.iff_injective_and_split hf,
    ← Module.Projective.iff_split_of_projective]
  /-
    case hs
    R P S : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing P
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R P
    inst✝³ : Algebra P S
    inst✝² : Algebra R S
    inst✝¹ : IsScalarTower R P S
    hf : Function.Surjective ⇑(algebraMap P S)
    inst✝ : Algebra.FormallySmooth R P
    ⊢ Function.Surjective ⇑(KaehlerDifferential.mapBaseChange R P S)
  -/
  exact KaehlerDifferential.mapBaseChange_surjective _ _ _ hf
  /-
    🎉 no goals
  -/


/--
An algebra is formally smooth if and only if `H¹(L_{R/S}) = 0` and `Ω_{S/R}` is projective.
-/
@[stacks 031J]
theorem Algebra.FormallySmooth.iff_subsingleton_and_projective :
    Algebra.FormallySmooth R S ↔
        Subsingleton (Algebra.H1Cotangent R S) ∧ Module.Projective S (Ω[S⁄R]) := by
  refine (Algebra.FormallySmooth.iff_injective_and_projective
    (Generators.self R S).algebraMap_surjective).trans (and_congr ?_ Iff.rfl)
  /-
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Iff (Function.Injective ⇑(KaehlerDifferential.kerCotangentToTensor R (Algebr …
  -/
  show Function.Injective (Generators.self R S).toExtension.cotangentComplex ↔ _
  /-
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Iff (Function.Injective ⇑(Algebra.Generators.self R S).toExtension.cotangent …
  -/
  rw [← LinearMap.ker_eq_bot, ← Submodule.subsingleton_iff_eq_bot]
  /-
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Iff (Subsingleton (Subtype fun x => Membership.mem (LinearMap.ker (Algebra.G …
  -/
  rfl
  /-
    🎉 no goals
  -/

