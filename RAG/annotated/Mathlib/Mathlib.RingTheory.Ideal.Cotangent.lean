/-- `I ⧸ I ^ 2` as a quotient of `I`. -/
def Cotangent : Type _ := I ⧸ (I • ⊤ : Submodule R I)


                                          /-
                                            R : Type u
                                            S : Type v
                                            S' : Type w
                                            inst✝⁶ : CommRing R
                                            inst✝⁵ : CommSemiring S
                                            inst✝⁴ : Algebra S R
                                            inst✝³ : CommSemiring S'
                                            inst✝² : Algebra S' R
                                            inst✝¹ : Algebra S S'
                                            inst✝ : IsScalarTower S S' R
                                            I : Ideal R
                                            ⊢ AddCommGroup I.Cotangent
                                          -/
instance : AddCommGroup I.Cotangent := by delta Cotangent; infer_instance
                                                           /-
                                                             🎉 no goals
                                                           -/


                                                            /-
                                                              R : Type u
                                                              S : Type v
                                                              S' : Type w
                                                              inst✝⁶ : CommRing R
                                                              inst✝⁵ : CommSemiring S
                                                              inst✝⁴ : Algebra S R
                                                              inst✝³ : CommSemiring S'
                                                              inst✝² : Algebra S' R
                                                              inst✝¹ : Algebra S S'
                                                              inst✝ : IsScalarTower S S' R
                                                              I : Ideal R
                                                              ⊢ Module (HasQuotient.Quotient R I) I.Cotangent
                                                            -/
instance cotangentModule : Module (R ⧸ I) I.Cotangent := by delta Cotangent; infer_instance
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


instance : Inhabited I.Cotangent := ⟨0⟩


instance Cotangent.moduleOfTower : Module S I.Cotangent :=
  Submodule.Quotient.module' _


instance Cotangent.isScalarTower : IsScalarTower S S' I.Cotangent :=
  Submodule.Quotient.isScalarTower _ _


instance [IsNoetherian R I] : IsNoetherian R I.Cotangent :=
  inferInstanceAs (IsNoetherian R (I ⧸ (I • ⊤ : Submodule R I)))


/-- The quotient map from `I` to `I ⧸ I ^ 2`. -/
@[simps! (config := .lemmasOnly) apply]
def toCotangent : I →ₗ[R] I.Cotangent := Submodule.mkQ _


theorem map_toCotangent_ker : I.toCotangent.ker.map I.subtype = I ^ 2 := by
  rw [Ideal.toCotangent, Submodule.ker_mkQ, pow_two, Submodule.map_smul'' I ⊤ (Submodule.subtype I),
    Algebra.id.smul_eq_mul, Submodule.map_subtype_top]


theorem mem_toCotangent_ker {x : I} : x ∈ LinearMap.ker I.toCotangent ↔ (x : R) ∈ I ^ 2 := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x : Subtype fun x => Membership.mem I x
    ⊢ Iff (Membership.mem (LinearMap.ker I.toCotangent) x) (Membership.mem (HPow.h …
  -/
  rw [← I.map_toCotangent_ker]
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x : Subtype fun x => Membership.mem I x
    ⊢ Iff (Membership.mem (LinearMap.ker I.toCotangent) x) (Membership.mem (Submod …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem toCotangent_eq {x y : I} : I.toCotangent x = I.toCotangent y ↔ (x - y : R) ∈ I ^ 2 := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x y : Subtype fun x => Membership.mem I x
    ⊢ Iff (Eq (I.toCotangent x) (I.toCotangent y)) (Membership.mem (HPow.hPow I 2) …
  -/
  rw [← sub_eq_zero]
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x y : Subtype fun x => Membership.mem I x
    ⊢ Iff (Eq (HSub.hSub (I.toCotangent x) (I.toCotangent y)) 0) (Membership.mem ( …
  -/
  exact I.mem_toCotangent_ker
  /-
    🎉 no goals
  -/


theorem toCotangent_eq_zero (x : I) : I.toCotangent x = 0 ↔ (x : R) ∈ I ^ 2 := I.mem_toCotangent_ker


theorem toCotangent_surjective : Function.Surjective I.toCotangent := Submodule.mkQ_surjective _


theorem toCotangent_range : LinearMap.range I.toCotangent = ⊤ := Submodule.range_mkQ _


theorem cotangent_subsingleton_iff : Subsingleton I.Cotangent ↔ IsIdempotentElem I := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Iff (Subsingleton I.Cotangent) (IsIdempotentElem I)
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      ⊢ Subsingleton I.Cotangent → IsIdempotentElem I
    -/
  · intro H
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      H : Subsingleton I.Cotangent
      ⊢ IsIdempotentElem I
    -/
    refine (pow_two I).symm.trans (le_antisymm (Ideal.pow_le_self two_ne_zero) ?_)
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      H : Subsingleton I.Cotangent
      ⊢ LE.le I (HPow.hPow I 2)
    -/
    exact fun x hx => (I.toCotangent_eq_zero ⟨x, hx⟩).mp (Subsingleton.elim _ _)
    /-
      🎉 no goals
    -/
  · exact fun e =>
      ⟨fun x y =>
        Quotient.inductionOn₂' x y fun x y =>
          I.toCotangent_eq.mpr <| ((pow_two I).trans e).symm ▸ I.sub_mem x.prop y.prop⟩


/-- The inclusion map `I ⧸ I ^ 2` to `R ⧸ I ^ 2`. -/
def cotangentToQuotientSquare : I.Cotangent →ₗ[R] R ⧸ I ^ 2 :=
  Submodule.mapQ (I • ⊤) (I ^ 2) I.subtype
    (by
      rw [← Submodule.map_le_iff_le_comap, Submodule.map_smul'', Submodule.map_top,
        Submodule.range_subtype, smul_eq_mul, pow_two] )


theorem to_quotient_square_comp_toCotangent :
    I.cotangentToQuotientSquare.comp I.toCotangent = (I ^ 2).mkQ.comp (Submodule.subtype I) :=
  LinearMap.ext fun _ => rfl


@[simp]
theorem toCotangent_to_quotient_square (x : I) :
    I.cotangentToQuotientSquare (I.toCotangent x) = (I ^ 2).mkQ x := rfl


lemma Cotangent.smul_eq_zero_of_mem {I : Ideal R}
    {x} (hx : x ∈ I) (m : I.Cotangent) : x • m = 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x : R
    hx : Membership.mem I x
    m : I.Cotangent
    ⊢ Eq (HSMul.hSMul x m) 0
  -/
  obtain ⟨m, rfl⟩ := Ideal.toCotangent_surjective _ m
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x : R
    hx : Membership.mem I x
    m : Subtype fun x => Membership.mem I x
    ⊢ Eq (HSMul.hSMul x (I.toCotangent m)) 0
  -/
  rw [← map_smul, Ideal.toCotangent_eq_zero, pow_two]
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x : R
    hx : Membership.mem I x
    m : Subtype fun x => Membership.mem I x
    ⊢ Membership.mem (HMul.hMul I I) ↑(HSMul.hSMul x m)
  -/
  exact Ideal.mul_mem_mul hx m.2
  /-
    🎉 no goals
  -/


lemma isTorsionBySet_cotangent :
    Module.IsTorsionBySet R I.Cotangent I :=
  fun m x ↦ m.smul_eq_zero_of_mem x.2


/-- `I ⧸ I ^ 2` as an ideal of `R ⧸ I ^ 2`. -/
def cotangentIdeal (I : Ideal R) : Ideal (R ⧸ I ^ 2) :=
  Submodule.map (Quotient.mk (I ^ 2)|>.toSemilinearMap) I


theorem cotangentIdeal_square (I : Ideal R) : I.cotangentIdeal ^ 2 = ⊥ := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Eq (HPow.hPow I.cotangentIdeal 2) Bot.bot
  -/
  rw [eq_bot_iff, pow_two I.cotangentIdeal, ← smul_eq_mul]
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    ⊢ LE.le (HSMul.hSMul I.cotangentIdeal I.cotangentIdeal) Bot.bot
  -/
  intro x hx
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x : HasQuotient.Quotient R (HPow.hPow I 2)
    hx : Membership.mem (HSMul.hSMul I.cotangentIdeal I.cotangentIdeal) x
    ⊢ Membership.mem Bot.bot x
  -/
  refine Submodule.smul_induction_on hx ?_ ?_
    /-
      case refine_1
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      x : HasQuotient.Quotient R (HPow.hPow I 2)
      hx : Membership.mem (HSMul.hSMul I.cotangentIdeal I.cotangentIdeal) x
      ⊢ ∀ (r : HasQuotient.Quotient R (HPow.hPow I 2)), Membership.mem I.cotangentId …
    -/
  · rintro _ ⟨x, hx, rfl⟩ _ ⟨y, hy, rfl⟩; apply (Submodule.Quotient.eq _).mpr _
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      x✝ : HasQuotient.Quotient R (HPow.hPow I 2)
      hx✝ : Membership.mem (HSMul.hSMul I.cotangentIdeal I.cotangentIdeal) x✝
      x : R
      hx : Membership.mem (↑I) x
      y : R
      hy : Membership.mem (↑I) y
      ⊢ Membership.mem (HPow.hPow I 2) (HSub.hSub ((fun x1 x2 => HMul.hMul x1 x2) x  …
    -/
    rw [sub_zero, pow_two]; exact Ideal.mul_mem_mul hx hy
                            /-
                              🎉 no goals
                            -/
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      x : HasQuotient.Quotient R (HPow.hPow I 2)
      hx : Membership.mem (HSMul.hSMul I.cotangentIdeal I.cotangentIdeal) x
      ⊢ ∀ (x y : HasQuotient.Quotient R (HPow.hPow I 2)), Membership.mem Bot.bot x → …
    -/
  · intro x y hx hy; exact add_mem hx hy
                     /-
                       🎉 no goals
                     -/


theorem to_quotient_square_range :
    LinearMap.range I.cotangentToQuotientSquare = I.cotangentIdeal.restrictScalars R := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Eq (LinearMap.range I.cotangentToQuotientSquare) (Submodule.restrictScalars  …
  -/
  trans LinearMap.range (I.cotangentToQuotientSquare.comp I.toCotangent)
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      ⊢ Eq (LinearMap.range I.cotangentToQuotientSquare) (LinearMap.range (I.cotange …
    -/
  · rw [LinearMap.range_comp, I.toCotangent_range, Submodule.map_top]
    /-
      🎉 no goals
    -/
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      ⊢ Eq (LinearMap.range (I.cotangentToQuotientSquare.comp I.toCotangent)) (Submo …
    -/
  · rw [to_quotient_square_comp_toCotangent, LinearMap.range_comp, I.range_subtype]; ext; rfl
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


/-- The equivalence of the two definitions of `I / I ^ 2`, either as the quotient of `I` or the
ideal of `R / I ^ 2`. -/
noncomputable def cotangentEquivIdeal : I.Cotangent ≃ₗ[R] I.cotangentIdeal := by
  refine
  { LinearMap.codRestrict (I.cotangentIdeal.restrictScalars R) I.cotangentToQuotientSquare
      fun x => by { rw [← to_quotient_square_range]; exact LinearMap.mem_range_self _ _ },
    Equiv.ofBijective _ ⟨?_, ?_⟩ with }
    /-
      case refine_1
      R : Type u
      S : Type v
      S' : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra S R
      inst✝³ : CommSemiring S'
      inst✝² : Algebra S' R
      inst✝¹ : Algebra S S'
      inst✝ : IsScalarTower S S' R
      I : Ideal R
      ⊢ Function.Injective fun c => ⟨I.cotangentToQuotientSquare c, ⋯⟩
    -/
  · rintro x y e
    /-
      case refine_1
      R : Type u
      S : Type v
      S' : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra S R
      inst✝³ : CommSemiring S'
      inst✝² : Algebra S' R
      inst✝¹ : Algebra S S'
      inst✝ : IsScalarTower S S' R
      I : Ideal R
      x y : I.Cotangent
      e : Eq ((fun c => ⟨I.cotangentToQuotientSquare c, ⋯⟩) x) ((fun c => ⟨I.cotange …
      ⊢ Eq x y
    -/
    replace e := congr_arg Subtype.val e
    /-
      case refine_1
      R : Type u
      S : Type v
      S' : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra S R
      inst✝³ : CommSemiring S'
      inst✝² : Algebra S' R
      inst✝¹ : Algebra S S'
      inst✝ : IsScalarTower S S' R
      I : Ideal R
      x y : I.Cotangent
      e : Eq ↑((fun c => ⟨I.cotangentToQuotientSquare c, ⋯⟩) x) ↑((fun c => ⟨I.cotan …
      ⊢ Eq x y
    -/
    obtain ⟨x, rfl⟩ := I.toCotangent_surjective x
    /-
      case refine_1.intro
      R : Type u
      S : Type v
      S' : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra S R
      inst✝³ : CommSemiring S'
      inst✝² : Algebra S' R
      inst✝¹ : Algebra S S'
      inst✝ : IsScalarTower S S' R
      I : Ideal R
      y : I.Cotangent
      x : Subtype fun x => Membership.mem I x
      e : Eq ↑((fun c => ⟨I.cotangentToQuotientSquare c, ⋯⟩) (I.toCotangent x)) ↑((f …
      ⊢ Eq (I.toCotangent x) y
    -/
    obtain ⟨y, rfl⟩ := I.toCotangent_surjective y
    /-
      case refine_1.intro.intro
      R : Type u
      S : Type v
      S' : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra S R
      inst✝³ : CommSemiring S'
      inst✝² : Algebra S' R
      inst✝¹ : Algebra S S'
      inst✝ : IsScalarTower S S' R
      I : Ideal R
      x y : Subtype fun x => Membership.mem I x
      e : Eq ↑((fun c => ⟨I.cotangentToQuotientSquare c, ⋯⟩) (I.toCotangent x)) ↑((f …
      ⊢ Eq (I.toCotangent x) (I.toCotangent y)
    -/
    rw [I.toCotangent_eq]
    /-
      case refine_1.intro.intro
      R : Type u
      S : Type v
      S' : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra S R
      inst✝³ : CommSemiring S'
      inst✝² : Algebra S' R
      inst✝¹ : Algebra S S'
      inst✝ : IsScalarTower S S' R
      I : Ideal R
      x y : Subtype fun x => Membership.mem I x
      e : Eq ↑((fun c => ⟨I.cotangentToQuotientSquare c, ⋯⟩) (I.toCotangent x)) ↑((f …
      ⊢ Membership.mem (HPow.hPow I 2) (HSub.hSub ↑x ↑y)
    -/
    dsimp only [toCotangent_to_quotient_square, Submodule.mkQ_apply] at e
    /-
      case refine_1.intro.intro
      R : Type u
      S : Type v
      S' : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra S R
      inst✝³ : CommSemiring S'
      inst✝² : Algebra S' R
      inst✝¹ : Algebra S S'
      inst✝ : IsScalarTower S S' R
      I : Ideal R
      x y : Subtype fun x => Membership.mem I x
      e : Eq (Submodule.Quotient.mk ↑x) (Submodule.Quotient.mk ↑y)
      ⊢ Membership.mem (HPow.hPow I 2) (HSub.hSub ↑x ↑y)
    -/
    rwa [Submodule.Quotient.eq] at e
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      S : Type v
      S' : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra S R
      inst✝³ : CommSemiring S'
      inst✝² : Algebra S' R
      inst✝¹ : Algebra S S'
      inst✝ : IsScalarTower S S' R
      I : Ideal R
      ⊢ Function.Surjective fun c => ⟨I.cotangentToQuotientSquare c, ⋯⟩
    -/
  · rintro ⟨_, x, hx, rfl⟩
    /-
      case refine_2.mk.intro.intro
      R : Type u
      S : Type v
      S' : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra S R
      inst✝³ : CommSemiring S'
      inst✝² : Algebra S' R
      inst✝¹ : Algebra S S'
      inst✝ : IsScalarTower S S' R
      I : Ideal R
      x : R
      hx : Membership.mem (↑I) x
      ⊢ Exists fun a => Eq ((fun c => ⟨I.cotangentToQuotientSquare c, ⋯⟩) a) ⟨(Ideal …
    -/
    exact ⟨I.toCotangent ⟨x, hx⟩, Subtype.ext rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem cotangentEquivIdeal_apply (x : I.Cotangent) :
    ↑(I.cotangentEquivIdeal x) = I.cotangentToQuotientSquare x := rfl


theorem cotangentEquivIdeal_symm_apply (x : R) (hx : x ∈ I) :
    -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specify `(R₂ := R)` because `I.toCotangent` suggested `R ⧸ I^2` instead
    I.cotangentEquivIdeal.symm ⟨(I ^ 2).mkQ x,
      -- timeout (200000 heartbeats) without `by exact`
         /-
           R : Type u
           S : Type v
           S' : Type w
           inst✝⁶ : CommRing R
           inst✝⁵ : CommSemiring S
           inst✝⁴ : Algebra S R
           inst✝³ : CommSemiring S'
           inst✝² : Algebra S' R
           inst✝¹ : Algebra S S'
           inst✝ : IsScalarTower S S' R
           I : Ideal R
           x : R
           hx : Membership.mem I x
           ⊢ Membership.mem I.cotangentIdeal ((Submodule.mkQ (HPow.hPow I 2)) x)
         -/
      by exact Submodule.mem_map_of_mem (F := R →ₗ[R] R ⧸ I ^ 2) (f := (I ^ 2).mkQ) hx⟩ =
         /-
           🎉 no goals
         -/
      I.toCotangent (R := R) ⟨x, hx⟩ := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x : R
    hx : Membership.mem I x
    ⊢ Eq (I.cotangentEquivIdeal.symm ⟨(Submodule.mkQ (HPow.hPow I 2)) x, ⋯⟩) (I.to …
  -/
  apply I.cotangentEquivIdeal.injective
  /-
    case a
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x : R
    hx : Membership.mem I x
    ⊢ Eq (I.cotangentEquivIdeal (I.cotangentEquivIdeal.symm ⟨(Submodule.mkQ (HPow. …
  -/
  rw [I.cotangentEquivIdeal.apply_symm_apply]
  /-
    case a
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x : R
    hx : Membership.mem I x
    ⊢ Eq ⟨(Submodule.mkQ (HPow.hPow I 2)) x, ⋯⟩ (I.cotangentEquivIdeal (I.toCotang …
  -/
  ext
  /-
    case a.a
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x : R
    hx : Membership.mem I x
    ⊢ Eq ↑⟨(Submodule.mkQ (HPow.hPow I 2)) x, ⋯⟩ ↑(I.cotangentEquivIdeal (I.toCota …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The lift of `f : A →ₐ[R] B` to `A ⧸ J ^ 2 →ₐ[R] B` with `J` being the kernel of `f`. -/
def _root_.AlgHom.kerSquareLift (f : A →ₐ[R] B) : A ⧸ RingHom.ker f.toRingHom ^ 2 →ₐ[R] B := by
  /-
    R : Type u
    S : Type v
    S' : Type w
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra S R
    inst✝⁷ : CommSemiring S'
    inst✝⁶ : Algebra S' R
    inst✝⁵ : Algebra S S'
    inst✝⁴ : IsScalarTower S S' R
    I : Ideal R
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f : AlgHom R A B
    ⊢ AlgHom R (HasQuotient.Quotient A (HPow.hPow (RingHom.ker f.toRingHom) 2)) B
  -/
  refine { Ideal.Quotient.lift (RingHom.ker f.toRingHom ^ 2) f.toRingHom ?_ with commutes' := ?_ }
    /-
      case refine_1
      R : Type u
      S : Type v
      S' : Type w
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : CommSemiring S'
      inst✝⁶ : Algebra S' R
      inst✝⁵ : Algebra S S'
      inst✝⁴ : IsScalarTower S S' R
      I : Ideal R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      ⊢ ∀ (a : A), Membership.mem (HPow.hPow (RingHom.ker f.toRingHom) 2) a → Eq (f. …
    -/
  · intro a ha; exact Ideal.pow_le_self two_ne_zero ha
                /-
                  🎉 no goals
                -/
    /-
      case refine_2
      R : Type u
      S : Type v
      S' : Type w
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : CommSemiring S'
      inst✝⁶ : Algebra S' R
      inst✝⁵ : Algebra S S'
      inst✝⁴ : IsScalarTower S S' R
      I : Ideal R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      ⊢ ∀ (r : R), Eq ((↑↑__src✝).toFun ((algebraMap R (HasQuotient.Quotient A (HPow …
    -/
  · intro r
    rw [IsScalarTower.algebraMap_apply R A, RingHom.toFun_eq_coe, Ideal.Quotient.algebraMap_eq,
      Ideal.Quotient.lift_mk]
    /-
      case refine_2
      R : Type u
      S : Type v
      S' : Type w
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : CommSemiring S'
      inst✝⁶ : Algebra S' R
      inst✝⁵ : Algebra S S'
      inst✝⁴ : IsScalarTower S S' R
      I : Ideal R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      r : R
      ⊢ Eq (f.toRingHom ((algebraMap R A) r)) ((algebraMap R B) r)
    -/
    exact f.map_algebraMap r
    /-
      🎉 no goals
    -/


theorem _root_.AlgHom.ker_kerSquareLift (f : A →ₐ[R] B) :
    RingHom.ker f.kerSquareLift.toRingHom = f.toRingHom.ker.cotangentIdeal := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f : AlgHom R A B
    ⊢ Eq (RingHom.ker f.kerSquareLift.toRingHom) (RingHom.ker f.toRingHom).cotange …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      inst✝⁴ : CommRing R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      ⊢ LE.le (RingHom.ker f.kerSquareLift.toRingHom) (RingHom.ker f.toRingHom).cota …
    -/
  · intro x hx; obtain ⟨x, rfl⟩ := Ideal.Quotient.mk_surjective x; exact ⟨x, hx, rfl⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    /-
      case a
      R : Type u
      inst✝⁴ : CommRing R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      ⊢ LE.le (RingHom.ker f.toRingHom).cotangentIdeal (RingHom.ker f.kerSquareLift. …
    -/
  · rintro _ ⟨x, hx, rfl⟩; exact hx
                           /-
                             🎉 no goals
                           -/


instance Algebra.kerSquareLift : Algebra (R ⧸ (RingHom.ker (algebraMap R A) ^ 2)) A :=
  (Algebra.ofId R A).kerSquareLift.toAlgebra


instance [Algebra A B] [IsScalarTower R A B] :
    IsScalarTower R (A ⧸ (RingHom.ker (algebraMap A B) ^ 2)) B :=
  IsScalarTower.of_algebraMap_eq'
    (IsScalarTower.toAlgHom R A B).kerSquareLift.comp_algebraMap.symm


/-- The quotient ring of `I ⧸ I ^ 2` is `R ⧸ I`. -/
def quotCotangent : (R ⧸ I ^ 2) ⧸ I.cotangentIdeal ≃+* R ⧸ I := by
  /-
    R : Type u
    S : Type v
    S' : Type w
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra S R
    inst✝⁷ : CommSemiring S'
    inst✝⁶ : Algebra S' R
    inst✝⁵ : Algebra S S'
    inst✝⁴ : IsScalarTower S S' R
    I : Ideal R
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    ⊢ RingEquiv (HasQuotient.Quotient (HasQuotient.Quotient R (HPow.hPow I 2)) I.c …
  -/
  refine (Ideal.quotEquivOfEq (Ideal.map_eq_submodule_map _ _).symm).trans ?_
  /-
    R : Type u
    S : Type v
    S' : Type w
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra S R
    inst✝⁷ : CommSemiring S'
    inst✝⁶ : Algebra S' R
    inst✝⁵ : Algebra S S'
    inst✝⁴ : IsScalarTower S S' R
    I : Ideal R
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    ⊢ RingEquiv (HasQuotient.Quotient (HasQuotient.Quotient R (HPow.hPow I 2)) (Id …
  -/
  refine (DoubleQuot.quotQuotEquivQuotSup _ _).trans ?_
  /-
    R : Type u
    S : Type v
    S' : Type w
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra S R
    inst✝⁷ : CommSemiring S'
    inst✝⁶ : Algebra S' R
    inst✝⁵ : Algebra S S'
    inst✝⁴ : IsScalarTower S S' R
    I : Ideal R
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    ⊢ RingEquiv (HasQuotient.Quotient R (Max.max (HPow.hPow I 2) I)) (HasQuotient. …
  -/
  exact Ideal.quotEquivOfEq (sup_eq_right.mpr <| Ideal.pow_le_self two_ne_zero)
  /-
    🎉 no goals
  -/


/-- The map `I/I² → J/J²` if `I ≤ f⁻¹(J)`. -/
def mapCotangent (I₁ : Ideal A) (I₂ : Ideal B) (f : A →ₐ[R] B) (h : I₁ ≤ I₂.comap f) :
    I₁.Cotangent →ₗ[R] I₂.Cotangent := by
  refine Submodule.mapQ ((I₁ • ⊤ : Submodule A I₁).restrictScalars R)
    ((I₂ • ⊤ : Submodule B I₂).restrictScalars R) ?_ ?_
    /-
      case refine_1
      R : Type u
      S : Type v
      S' : Type w
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : CommSemiring S'
      inst✝⁶ : Algebra S' R
      inst✝⁵ : Algebra S S'
      inst✝⁴ : IsScalarTower S S' R
      I : Ideal R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I₁ : Ideal A
      I₂ : Ideal B
      f : AlgHom R A B
      h : LE.le I₁ (Ideal.comap f I₂)
      ⊢ LinearMap (RingHom.id R) (Subtype fun x => Membership.mem I₁ x) (Subtype fun …
    -/
  · exact f.toLinearMap.restrict (p := I₁.restrictScalars R) (q := I₂.restrictScalars R) h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      S : Type v
      S' : Type w
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : CommSemiring S'
      inst✝⁶ : Algebra S' R
      inst✝⁵ : Algebra S S'
      inst✝⁴ : IsScalarTower S S' R
      I : Ideal R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I₁ : Ideal A
      I₂ : Ideal B
      f : AlgHom R A B
      h : LE.le I₁ (Ideal.comap f I₂)
      ⊢ LE.le (Submodule.restrictScalars R (HSMul.hSMul I₁ Top.top)) (Submodule.coma …
    -/
  · intro x hx
    /-
      case refine_2
      R : Type u
      S : Type v
      S' : Type w
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : CommSemiring S'
      inst✝⁶ : Algebra S' R
      inst✝⁵ : Algebra S S'
      inst✝⁴ : IsScalarTower S S' R
      I : Ideal R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I₁ : Ideal A
      I₂ : Ideal B
      f : AlgHom R A B
      h : LE.le I₁ (Ideal.comap f I₂)
      x : Subtype fun x => Membership.mem I₁ x
      hx : Membership.mem (Submodule.restrictScalars R (HSMul.hSMul I₁ Top.top)) x
      ⊢ Membership.mem (Submodule.comap (f.toLinearMap.restrict h) (Submodule.restri …
    -/
    rw [Submodule.restrictScalars_mem] at hx
    /-
      case refine_2
      R : Type u
      S : Type v
      S' : Type w
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : CommSemiring S'
      inst✝⁶ : Algebra S' R
      inst✝⁵ : Algebra S S'
      inst✝⁴ : IsScalarTower S S' R
      I : Ideal R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I₁ : Ideal A
      I₂ : Ideal B
      f : AlgHom R A B
      h : LE.le I₁ (Ideal.comap f I₂)
      x : Subtype fun x => Membership.mem I₁ x
      hx : Membership.mem (HSMul.hSMul I₁ Top.top) x
      ⊢ Membership.mem (Submodule.comap (f.toLinearMap.restrict h) (Submodule.restri …
    -/
    refine Submodule.smul_induction_on hx ?_ (fun _ _ ↦ add_mem)
    /-
      case refine_2
      R : Type u
      S : Type v
      S' : Type w
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : CommSemiring S'
      inst✝⁶ : Algebra S' R
      inst✝⁵ : Algebra S S'
      inst✝⁴ : IsScalarTower S S' R
      I : Ideal R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I₁ : Ideal A
      I₂ : Ideal B
      f : AlgHom R A B
      h : LE.le I₁ (Ideal.comap f I₂)
      x : Subtype fun x => Membership.mem I₁ x
      hx : Membership.mem (HSMul.hSMul I₁ Top.top) x
      ⊢ ∀ (r : A), Membership.mem I₁ r → ∀ (n : Subtype fun x => Membership.mem I₁ x …
    -/
    rintro a ha ⟨b, hb⟩ -
    /-
      case refine_2.mk
      R : Type u
      S : Type v
      S' : Type w
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : CommSemiring S'
      inst✝⁶ : Algebra S' R
      inst✝⁵ : Algebra S S'
      inst✝⁴ : IsScalarTower S S' R
      I : Ideal R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I₁ : Ideal A
      I₂ : Ideal B
      f : AlgHom R A B
      h : LE.le I₁ (Ideal.comap f I₂)
      x : Subtype fun x => Membership.mem I₁ x
      hx : Membership.mem (HSMul.hSMul I₁ Top.top) x
      a : A
      ha : Membership.mem I₁ a
      b : A
      hb : Membership.mem I₁ b
      ⊢ Membership.mem (Submodule.comap (f.toLinearMap.restrict h) (Submodule.restri …
    -/
    simp only [SetLike.mk_smul_mk, smul_eq_mul, Submodule.mem_comap, Submodule.restrictScalars_mem]
    convert (Submodule.smul_mem_smul (M := I₂) (r := f a)
      (n := ⟨f b, h hb⟩) (h ha) (Submodule.mem_top)) using 1
    /-
      case h.e'_5
      R : Type u
      S : Type v
      S' : Type w
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : CommSemiring S'
      inst✝⁶ : Algebra S' R
      inst✝⁵ : Algebra S S'
      inst✝⁴ : IsScalarTower S S' R
      I : Ideal R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I₁ : Ideal A
      I₂ : Ideal B
      f : AlgHom R A B
      h : LE.le I₁ (Ideal.comap f I₂)
      x : Subtype fun x => Membership.mem I₁ x
      hx : Membership.mem (HSMul.hSMul I₁ Top.top) x
      a : A
      ha : Membership.mem I₁ a
      b : A
      hb : Membership.mem I₁ b
      ⊢ Eq ((f.toLinearMap.restrict h) ⟨HMul.hMul a b, ⋯⟩) (HSMul.hSMul (f a) ⟨f b,  …
    -/
    ext
    /-
      case h.e'_5.a
      R : Type u
      S : Type v
      S' : Type w
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : CommSemiring S'
      inst✝⁶ : Algebra S' R
      inst✝⁵ : Algebra S S'
      inst✝⁴ : IsScalarTower S S' R
      I : Ideal R
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I₁ : Ideal A
      I₂ : Ideal B
      f : AlgHom R A B
      h : LE.le I₁ (Ideal.comap f I₂)
      x : Subtype fun x => Membership.mem I₁ x
      hx : Membership.mem (HSMul.hSMul I₁ Top.top) x
      a : A
      ha : Membership.mem I₁ a
      b : A
      hb : Membership.mem I₁ b
      ⊢ Eq ↑((f.toLinearMap.restrict h) ⟨HMul.hMul a b, ⋯⟩) ↑(HSMul.hSMul (f a) ⟨f b …
    -/
    exact _root_.map_mul f a b
    /-
      🎉 no goals
    -/


@[simp]
lemma mapCotangent_toCotangent
    (I₁ : Ideal A) (I₂ : Ideal B) (f : A →ₐ[R] B) (h : I₁ ≤ I₂.comap f) (x : I₁) :
    Ideal.mapCotangent I₁ I₂ f h (Ideal.toCotangent I₁ x) = Ideal.toCotangent I₂ ⟨f x, h x.2⟩ := rfl


/-- The `A ⧸ I`-vector space `I ⧸ I ^ 2`. -/
abbrev CotangentSpace : Type _ := (maximalIdeal R).Cotangent


instance : Module (ResidueField R) (CotangentSpace R) := Ideal.cotangentModule _


instance : IsScalarTower R (ResidueField R) (CotangentSpace R) :=
  Module.IsTorsionBySet.isScalarTower _


instance [IsNoetherianRing R] : FiniteDimensional (ResidueField R) (CotangentSpace R) :=
  Module.Finite.of_restrictScalars_finite R _ _


lemma subsingleton_cotangentSpace_iff [IsNoetherianRing R] :
    Subsingleton (CotangentSpace R) ↔ IsField R := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsNoetherianRing R
    ⊢ Iff (Subsingleton (IsLocalRing.CotangentSpace R)) (IsField R)
  -/
  refine (maximalIdeal R).cotangent_subsingleton_iff.trans ?_
  rw [IsLocalRing.isField_iff_maximalIdeal_eq,
    Ideal.isIdempotentElem_iff_eq_bot_or_top_of_isLocalRing]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsNoetherianRing R
    ⊢ Iff (Or (Eq (IsLocalRing.maximalIdeal R) Bot.bot) (Eq (IsLocalRing.maximalId …
  -/
  simp [(maximalIdeal.isMaximal R).ne_top]
  /-
    🎉 no goals
  -/


lemma CotangentSpace.map_eq_top_iff [IsNoetherianRing R] {M : Submodule R (maximalIdeal R)} :
    M.map (maximalIdeal R).toCotangent = ⊤ ↔ M = ⊤ := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsNoetherianRing R
    M : Submodule R (Subtype fun x => Membership.mem (IsLocalRing.maximalIdeal R) x)
    ⊢ Iff (Eq (Submodule.map (IsLocalRing.maximalIdeal R).toCotangent M) Top.top)  …
  -/
  refine ⟨fun H ↦ eq_top_iff.mpr ?_, by rintro rfl; simp [Ideal.toCotangent_range]⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsNoetherianRing R
    M : Submodule R (Subtype fun x => Membership.mem (IsLocalRing.maximalIdeal R) x)
    H : Eq (Submodule.map (IsLocalRing.maximalIdeal R).toCotangent M) Top.top
    ⊢ LE.le Top.top M
  -/
  refine (Submodule.map_le_map_iff_of_injective (Submodule.injective_subtype _) _ _).mp ?_
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsNoetherianRing R
    M : Submodule R (Subtype fun x => Membership.mem (IsLocalRing.maximalIdeal R) x)
    H : Eq (Submodule.map (IsLocalRing.maximalIdeal R).toCotangent M) Top.top
    ⊢ LE.le (Submodule.map (Submodule.subtype (IsLocalRing.maximalIdeal R)) Top.to …
  -/
  rw [Submodule.map_top, Submodule.range_subtype]
  apply Submodule.le_of_le_smul_of_le_jacobson_bot (IsNoetherian.noetherian _)
    (IsLocalRing.jacobson_eq_maximalIdeal _ bot_ne_top).ge
  rw [smul_eq_mul, ← pow_two, ← Ideal.map_toCotangent_ker, ← Submodule.map_sup,
    ← Submodule.comap_map_eq, H, Submodule.comap_top, Submodule.map_top, Submodule.range_subtype]


lemma CotangentSpace.span_image_eq_top_iff [IsNoetherianRing R] {s : Set (maximalIdeal R)} :
    Submodule.span (ResidueField R) ((maximalIdeal R).toCotangent '' s) = ⊤ ↔
      Submodule.span R s = ⊤ := by
  rw [← map_eq_top_iff, ← (Submodule.restrictScalars_injective R ..).eq_iff,
    Submodule.restrictScalars_span]
    /-
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsLocalRing R
      inst✝ : IsNoetherianRing R
      s : Set (Subtype fun x => Membership.mem (IsLocalRing.maximalIdeal R) x)
      ⊢ Iff (Eq (Submodule.span R (Set.image (⇑(IsLocalRing.maximalIdeal R).toCotang …
    -/
  · simp only [Ideal.toCotangent_apply, Submodule.restrictScalars_top, Submodule.map_span]
    /-
      🎉 no goals
    -/
    /-
      case hsur
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsLocalRing R
      inst✝ : IsNoetherianRing R
      s : Set (Subtype fun x => Membership.mem (IsLocalRing.maximalIdeal R) x)
      ⊢ Function.Surjective ⇑(algebraMap R (IsLocalRing.ResidueField R))
    -/
  · exact Ideal.Quotient.mk_surjective
    /-
      🎉 no goals
    -/


lemma finrank_cotangentSpace_eq_zero_iff [IsNoetherianRing R] :
    finrank (ResidueField R) (CotangentSpace R) = 0 ↔ IsField R := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsNoetherianRing R
    ⊢ Iff (Eq (Module.finrank (IsLocalRing.ResidueField R) (IsLocalRing.CotangentS …
  -/
  rw [finrank_zero_iff, subsingleton_cotangentSpace_iff]
  /-
    🎉 no goals
  -/


lemma finrank_cotangentSpace_eq_zero (R) [Field R] :
    finrank (ResidueField R) (CotangentSpace R) = 0 :=
  finrank_cotangentSpace_eq_zero_iff.mpr (Field.toIsField R)


open Submodule in
theorem finrank_cotangentSpace_le_one_iff [IsNoetherianRing R] :
    finrank (ResidueField R) (CotangentSpace R) ≤ 1 ↔ (maximalIdeal R).IsPrincipal := by
  rw [Module.finrank_le_one_iff_top_isPrincipal, isPrincipal_iff,
    (maximalIdeal R).toCotangent_surjective.exists, isPrincipal_iff]
  simp_rw [← Set.image_singleton, eq_comm (a := ⊤), CotangentSpace.span_image_eq_top_iff,
    ← (map_injective_of_injective (injective_subtype _)).eq_iff, map_span, Set.image_singleton,
    Submodule.map_top, range_subtype, eq_comm (a := maximalIdeal R)]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsLocalRing R
    inst✝ : IsNoetherianRing R
    ⊢ Iff (Exists fun x => Eq (Submodule.span R (Singleton.singleton ((Submodule.s …
  -/
  exact ⟨fun ⟨x, h⟩ ↦ ⟨_, h⟩, fun ⟨x, h⟩ ↦ ⟨⟨x, h ▸ subset_span (Set.mem_singleton x)⟩, h⟩⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-11")]
alias LocalRing.CotangentSpace := IsLocalRing.CotangentSpace


@[deprecated (since := "2024-11-11")]
alias LocalRing.subsingleton_cotangentSpace_iff := IsLocalRing.subsingleton_cotangentSpace_iff


@[deprecated (since := "2024-11-11")]
alias LocalRing.map_eq_top_iff := IsLocalRing.CotangentSpace.map_eq_top_iff


@[deprecated (since := "2024-11-11")]
alias LocalRing.span_image_eq_top_iff := IsLocalRing.CotangentSpace.span_image_eq_top_iff


@[deprecated (since := "2024-11-11")]
alias LocalRing.finrank_cotangentSpace_eq_zero_iff := IsLocalRing.finrank_cotangentSpace_eq_zero_iff


@[deprecated (since := "2024-11-11")]
alias LocalRing.finrank_cotangentSpace_eq_zero := IsLocalRing.finrank_cotangentSpace_eq_zero


@[deprecated (since := "2024-11-11")]
alias LocalRing.finrank_cotangentSpace_le_one_iff := IsLocalRing.finrank_cotangentSpace_le_one_iff

