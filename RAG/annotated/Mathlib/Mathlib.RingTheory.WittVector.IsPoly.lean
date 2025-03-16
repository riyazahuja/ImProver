local notation "𝕎" => WittVector p -- type as `\bbW`


theorem poly_eq_of_wittPolynomial_bind_eq' [Fact p.Prime] (f g : ℕ → MvPolynomial (idx × ℕ) ℤ)
    (h : ∀ n, bind₁ f (wittPolynomial p _ n) = bind₁ g (wittPolynomial p _ n)) : f = g := by
  /-
    p : Nat
    idx : Type u_1
    inst✝ : Fact (Nat.Prime p)
    f g : Nat → MvPolynomial (Prod idx Nat) Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ f) (wittPolynomial p Int n)) ((MvPoly …
    ⊢ Eq f g
  -/
  ext1 n
  /-
    case h
    p : Nat
    idx : Type u_1
    inst✝ : Fact (Nat.Prime p)
    f g : Nat → MvPolynomial (Prod idx Nat) Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ f) (wittPolynomial p Int n)) ((MvPoly …
    n : Nat
    ⊢ Eq (f n) (g n)
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  /-
    case h.a
    p : Nat
    idx : Type u_1
    inst✝ : Fact (Nat.Prime p)
    f g : Nat → MvPolynomial (Prod idx Nat) Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ f) (wittPolynomial p Int n)) ((MvPoly …
    n : Nat
    ⊢ Eq ((MvPolynomial.map (Int.castRingHom Rat)) (f n)) ((MvPolynomial.map (Int. …
  -/
  rw [← funext_iff] at h
  replace h :=
    congr_arg (fun fam => bind₁ (MvPolynomial.map (Int.castRingHom ℚ) ∘ fam) (xInTermsOfW p ℚ n)) h
  simpa only [Function.comp_def, map_bind₁, map_wittPolynomial, ← bind₁_bind₁,
    bind₁_wittPolynomial_xInTermsOfW, bind₁_X_right] using h


theorem poly_eq_of_wittPolynomial_bind_eq [Fact p.Prime] (f g : ℕ → MvPolynomial ℕ ℤ)
    (h : ∀ n, bind₁ f (wittPolynomial p _ n) = bind₁ g (wittPolynomial p _ n)) : f = g := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : Nat → MvPolynomial Nat Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ f) (wittPolynomial p Int n)) ((MvPoly …
    ⊢ Eq f g
  -/
  ext1 n
  /-
    case h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : Nat → MvPolynomial Nat Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ f) (wittPolynomial p Int n)) ((MvPoly …
    n : Nat
    ⊢ Eq (f n) (g n)
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  /-
    case h.a
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : Nat → MvPolynomial Nat Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ f) (wittPolynomial p Int n)) ((MvPoly …
    n : Nat
    ⊢ Eq ((MvPolynomial.map (Int.castRingHom Rat)) (f n)) ((MvPolynomial.map (Int. …
  -/
  rw [← funext_iff] at h
  replace h :=
    congr_arg (fun fam => bind₁ (MvPolynomial.map (Int.castRingHom ℚ) ∘ fam) (xInTermsOfW p ℚ n)) h
  simpa only [Function.comp_def, map_bind₁, map_wittPolynomial, ← bind₁_bind₁,
    bind₁_wittPolynomial_xInTermsOfW, bind₁_X_right] using h

-- Ideally, we would generalise this to n-ary functions
-- But we don't have a good theory of n-ary compositions in mathlib

/--
A function `f : Π R, 𝕎 R → 𝕎 R` that maps Witt vectors to Witt vectors over arbitrary base rings
is said to be *polynomial* if there is a family of polynomials `φₙ` over `ℤ` such that the `n`th
coefficient of `f x` is given by evaluating `φₙ` at the coefficients of `x`.

See also `WittVector.IsPoly₂` for the binary variant.

The `ghost_calc` tactic makes use of the `IsPoly` and `IsPoly₂` typeclass and its instances.
(In Lean 3, there was an `@[is_poly]` attribute to manage these instances,
because typeclass resolution did not play well with function composition.
This no longer seems to be an issue, so that such instances can be defined directly.)
-/
class IsPoly (f : ∀ ⦃R⦄ [CommRing R], WittVector p R → 𝕎 R) : Prop where mk' ::
  poly :
    ∃ φ : ℕ → MvPolynomial ℕ ℤ,
      ∀ ⦃R⦄ [CommRing R] (x : 𝕎 R), (f x).coeff = fun n => aeval x.coeff (φ n)


/-- The identity function on Witt vectors is a polynomial function. -/
instance idIsPoly : IsPoly p fun _ _ => id :=
          /-
            p : Nat
            R S : Type u
            idx : Type u_1
            inst✝¹ : CommRing R
            inst✝ : CommRing S
            ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (id x).coeff f …
          -/
  ⟨⟨X, by intros; simp only [aeval_X, id]⟩⟩
                  /-
                    🎉 no goals
                  -/


instance idIsPolyI' : IsPoly p fun _ _ a => a :=
  WittVector.idIsPoly _


instance : Inhabited (IsPoly p fun _ _ => id) :=
  ⟨WittVector.idIsPoly p⟩


theorem ext [Fact p.Prime] {f g} (hf : IsPoly p f) (hg : IsPoly p g)
    (h : ∀ (R : Type u) [_Rcr : CommRing R] (x : 𝕎 R) (n : ℕ),
        ghostComponent n (f x) = ghostComponent n (g x)) :
    ∀ (R : Type u) [_Rcr : CommRing R] (x : 𝕎 R), f x = g x := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    hf : WittVector.IsPoly p f
    hg : WittVector.IsPoly p g
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    ⊢ ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R), Eq (f x) (g x)
  -/
  obtain ⟨φ, hf⟩ := hf
  /-
    case mk'.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    hg : WittVector.IsPoly p g
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ⊢ ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R), Eq (f x) (g x)
  -/
  obtain ⟨ψ, hg⟩ := hg
  /-
    case mk'.intro.mk'.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    ⊢ ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R), Eq (f x) (g x)
  -/
  intros
  /-
    case mk'.intro.mk'.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ : WittVector p R✝
    ⊢ Eq (f x✝) (g x✝)
  -/
  ext n
  /-
    case mk'.intro.mk'.intro.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ : WittVector p R✝
    n : Nat
    ⊢ Eq ((f x✝).coeff n) ((g x✝).coeff n)
  -/
  rw [hf, hg, poly_eq_of_wittPolynomial_bind_eq p φ ψ]
  /-
    case mk'.intro.mk'.intro.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ : WittVector p R✝
    n : Nat
    ⊢ ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPolyno …
  -/
  intro k
  /-
    case mk'.intro.mk'.intro.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ : WittVector p R✝
    n k : Nat
    ⊢ Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int k)) ((MvPolynomial.bind₁ ψ) …
  -/
  apply MvPolynomial.funext
  /-
    case mk'.intro.mk'.intro.h.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ : WittVector p R✝
    n k : Nat
    ⊢ ∀ (x : Nat → Int), Eq ((MvPolynomial.eval x) ((MvPolynomial.bind₁ φ) (wittPo …
  -/
  intro x
  /-
    case mk'.intro.mk'.intro.h.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ : WittVector p R✝
    n k : Nat
    x : Nat → Int
    ⊢ Eq ((MvPolynomial.eval x) ((MvPolynomial.bind₁ φ) (wittPolynomial p Int k))) …
  -/
  simp only [hom_bind₁]
  /-
    case mk'.intro.mk'.intro.h.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ : WittVector p R✝
    n k : Nat
    x : Nat → Int
    ⊢ Eq ((MvPolynomial.eval₂Hom ((MvPolynomial.eval x).comp MvPolynomial.C) fun i …
  -/
  specialize h (ULift ℤ) (mk p fun i => ⟨x i⟩) k
  /-
    case mk'.intro.mk'.intro.h.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ : WittVector p R✝
    n k : Nat
    x : Nat → Int
    h : Eq ((WittVector.ghostComponent k) (f (WittVector.mk p fun i => { down := x …
    ⊢ Eq ((MvPolynomial.eval₂Hom ((MvPolynomial.eval x).comp MvPolynomial.C) fun i …
  -/
  simp only [ghostComponent_apply, aeval_eq_eval₂Hom] at h
  /-
    case mk'.intro.mk'.intro.h.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ : WittVector p R✝
    n k : Nat
    x : Nat → Int
    h : Eq ((MvPolynomial.eval₂Hom (algebraMap Int (ULift.{u, 0} Int)) (f (WittVec …
    ⊢ Eq ((MvPolynomial.eval₂Hom ((MvPolynomial.eval x).comp MvPolynomial.C) fun i …
  -/
  apply (ULift.ringEquiv.symm : ℤ ≃+* _).injective
  /-
    case mk'.intro.mk'.intro.h.h.a
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ : WittVector p R✝
    n k : Nat
    x : Nat → Int
    h : Eq ((MvPolynomial.eval₂Hom (algebraMap Int (ULift.{u, 0} Int)) (f (WittVec …
    ⊢ Eq (ULift.ringEquiv.symm ((MvPolynomial.eval₂Hom ((MvPolynomial.eval x).comp …
  -/
  simp only [← RingEquiv.coe_toRingHom, map_eval₂Hom]
  /-
    case mk'.intro.mk'.intro.h.h.a
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff f …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ : WittVector p R✝
    n k : Nat
    x : Nat → Int
    h : Eq ((MvPolynomial.eval₂Hom (algebraMap Int (ULift.{u, 0} Int)) (f (WittVec …
    ⊢ Eq ((MvPolynomial.eval₂Hom ((↑ULift.ringEquiv.symm).comp ((MvPolynomial.eval …
  -/
  convert h using 1
  all_goals
    simp only [hf, hg, MvPolynomial.eval, map_eval₂Hom]
    apply eval₂Hom_congr (RingHom.ext_int _ _) _ rfl
    ext1
    apply eval₂Hom_congr (RingHom.ext_int _ _) _ rfl
    simp only [coeff_mk]; rfl


/-- The composition of polynomial functions is polynomial. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): made this an instance
instance comp {g f} [hg : IsPoly p g] [hf : IsPoly p f] :
    IsPoly p fun R _Rcr => @g R _Rcr ∘ @f R _Rcr := by
  /-
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    g f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    hg : WittVector.IsPoly p g
    hf : WittVector.IsPoly p f
    ⊢ WittVector.IsPoly p fun R _Rcr => Function.comp g f
  -/
  obtain ⟨φ, hf⟩ := hf
  /-
    case mk'.intro
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    g f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    hg : WittVector.IsPoly p g
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff …
    ⊢ WittVector.IsPoly p fun R _Rcr => Function.comp g f
  -/
  obtain ⟨ψ, hg⟩ := hg
  /-
    case mk'.intro.mk'.intro
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    g f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    ⊢ WittVector.IsPoly p fun R _Rcr => Function.comp g f
  -/
  use fun n => bind₁ φ (ψ n)
  /-
    case h
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    g f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (Function.comp …
  -/
  intros
  /-
    case h
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    g f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    R✝ : Type u_2
    inst✝ : CommRing R✝
    x✝ : WittVector p R✝
    ⊢ Eq (Function.comp g f x✝).coeff fun n => (MvPolynomial.aeval x✝.coeff) ((fun …
  -/
  simp only [aeval_bind₁, Function.comp, hg, hf]
  /-
    🎉 no goals
  -/


/-- A binary function `f : Π R, 𝕎 R → 𝕎 R → 𝕎 R` on Witt vectors
is said to be *polynomial* if there is a family of polynomials `φₙ` over `ℤ` such that the `n`th
coefficient of `f x y` is given by evaluating `φₙ` at the coefficients of `x` and `y`.

See also `WittVector.IsPoly` for the unary variant.

The `ghost_calc` tactic makes use of the `IsPoly` and `IsPoly₂` typeclass and its instances.
(In Lean 3, there was an `@[is_poly]` attribute to manage these instances,
because typeclass resolution did not play well with function composition.
This no longer seems to be an issue, so that such instances can be defined directly.)
-/
class IsPoly₂ (f : ∀ ⦃R⦄ [CommRing R], WittVector p R → 𝕎 R → 𝕎 R) : Prop where mk' ::
  poly :
    ∃ φ : ℕ → MvPolynomial (Fin 2 × ℕ) ℤ,
      ∀ ⦃R⦄ [CommRing R] (x y : 𝕎 R), (f x y).coeff = fun n => peval (φ n) ![x.coeff, y.coeff]


/-- The composition of polynomial functions is polynomial. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): made this an instance
instance IsPoly₂.comp {h f g} [hh : IsPoly₂ p h] [hf : IsPoly p f] [hg : IsPoly p g] :
    IsPoly₂ p fun _ _Rcr x y => h (f x) (g y) := by
  /-
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    h : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    f g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    hh : WittVector.IsPoly₂ p h
    hf : WittVector.IsPoly p f
    hg : WittVector.IsPoly p g
    ⊢ WittVector.IsPoly₂ p fun x _Rcr x_1 y => h (f x_1) (g y)
  -/
  obtain ⟨φ, hf⟩ := hf
  /-
    case mk'.intro
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    h : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    f g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    hh : WittVector.IsPoly₂ p h
    hg : WittVector.IsPoly p g
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff …
    ⊢ WittVector.IsPoly₂ p fun x _Rcr x_1 y => h (f x_1) (g y)
  -/
  obtain ⟨ψ, hg⟩ := hg
  /-
    case mk'.intro.mk'.intro
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    h : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    f g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    hh : WittVector.IsPoly₂ p h
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    ⊢ WittVector.IsPoly₂ p fun x _Rcr x_1 y => h (f x_1) (g y)
  -/
  obtain ⟨χ, hh⟩ := hh
  refine ⟨⟨fun n ↦ bind₁ (uncurry <|
    ![fun k ↦ rename (Prod.mk (0 : Fin 2)) (φ k),
      fun k ↦ rename (Prod.mk (1 : Fin 2)) (ψ k)]) (χ n), ?_⟩⟩
  /-
    case mk'.intro.mk'.intro.mk'.intro
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    h : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    f g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    χ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hh : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (h x y).c …
    ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (h (f x) (g  …
  -/
  intros
  /-
    case mk'.intro.mk'.intro.mk'.intro
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    h : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    f g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    χ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hh : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (h x y).c …
    R✝ : Type u_2
    inst✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    ⊢ Eq (h (f x✝) (g y✝)).coeff fun n => WittVector.peval ((fun n => (MvPolynomia …
  -/
  funext n
  simp (config := { unfoldPartialApp := true }) only [peval, aeval_bind₁, Function.comp, hh, hf, hg,
    uncurry]
  /-
    case mk'.intro.mk'.intro.mk'.intro.h
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    h : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    f g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    χ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hh : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (h x y).c …
    R✝ : Type u_2
    inst✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n : Nat
    ⊢ Eq ((MvPolynomial.aeval fun a => Matrix.vecCons (fun n => (MvPolynomial.aeva …
  -/
  apply eval₂Hom_congr rfl _ rfl
  /-
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    h : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    f g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    χ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hh : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (h x y).c …
    R✝ : Type u_2
    inst✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n : Nat
    ⊢ Eq (fun a => Matrix.vecCons (fun n => (MvPolynomial.aeval x✝.coeff) (φ n)) ( …
  -/
  ext ⟨i, n⟩
  /-
    case h.mk
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    h : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    f g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    χ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hh : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (h x y).c …
    R✝ : Type u_2
    inst✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n✝ : Nat
    i : Fin 2
    n : Nat
    ⊢ Eq (Matrix.vecCons (fun n => (MvPolynomial.aeval x✝.coeff) (φ n)) (Matrix.ve …
  -/
                  /-
                    🎉 no goals
                  -/
  fin_cases i <;> simp [aeval_eq_eval₂Hom, eval₂Hom_rename, Function.comp_def]
                  /-
                    🎉 no goals
                  -/


/-- The composition of a polynomial function with a binary polynomial function is polynomial. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): made this an instance
instance IsPoly.comp₂ {g f} [hg : IsPoly p g] [hf : IsPoly₂ p f] :
    IsPoly₂ p fun _ _Rcr x y => g (f x y) := by
  /-
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    hg : WittVector.IsPoly p g
    hf : WittVector.IsPoly₂ p f
    ⊢ WittVector.IsPoly₂ p fun x _Rcr x_1 y => g (f x_1 y)
  -/
  obtain ⟨φ, hf⟩ := hf
  /-
    case mk'.intro
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    hg : WittVector.IsPoly p g
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).c …
    ⊢ WittVector.IsPoly₂ p fun x _Rcr x_1 y => g (f x_1 y)
  -/
  obtain ⟨ψ, hg⟩ := hg
  /-
    case mk'.intro.mk'.intro
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).c …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    ⊢ WittVector.IsPoly₂ p fun x _Rcr x_1 y => g (f x_1 y)
  -/
  use fun n => bind₁ φ (ψ n)
  /-
    case h
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).c …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g (f x y)). …
  -/
  intros
  /-
    case h
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    g : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).c …
    ψ : Nat → MvPolynomial Nat Int
    hg : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (g x).coeff …
    R✝ : Type u_2
    inst✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    ⊢ Eq (g (f x✝ y✝)).coeff fun n => WittVector.peval ((fun n => (MvPolynomial.bi …
  -/
  simp only [peval, aeval_bind₁, Function.comp, hg, hf]
  /-
    🎉 no goals
  -/


/-- The diagonal `fun x ↦ f x x` of a polynomial function `f` is polynomial. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): made this an instance
instance IsPoly₂.diag {f} [hf : IsPoly₂ p f] : IsPoly p fun _ _Rcr x => f x x := by
  /-
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    hf : WittVector.IsPoly₂ p f
    ⊢ WittVector.IsPoly p fun x _Rcr x_1 => f x_1 x_1
  -/
  obtain ⟨φ, hf⟩ := hf
  /-
    case mk'.intro
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).c …
    ⊢ WittVector.IsPoly p fun x _Rcr x_1 => f x_1 x_1
  -/
  refine ⟨⟨fun n => bind₁ (uncurry ![X, X]) (φ n), ?_⟩⟩
  /-
    case mk'.intro
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).c …
    ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x x).coeff  …
  -/
  intros; funext n
  /-
    case mk'.intro.h
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).c …
    R✝ : Type u_2
    inst✝ : CommRing R✝
    x✝ : WittVector p R✝
    n : Nat
    ⊢ Eq ((f x✝ x✝).coeff n) ((MvPolynomial.aeval x✝.coeff) ((fun n => (MvPolynomi …
  -/
  simp (config := { unfoldPartialApp := true }) only [hf, peval, uncurry, aeval_bind₁]
  /-
    case mk'.intro.h
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).c …
    R✝ : Type u_2
    inst✝ : CommRing R✝
    x✝ : WittVector p R✝
    n : Nat
    ⊢ Eq ((MvPolynomial.aeval fun a => Matrix.vecCons x✝.coeff (Matrix.vecCons x✝. …
  -/
  apply eval₂Hom_congr rfl _ rfl
  /-
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).c …
    R✝ : Type u_2
    inst✝ : CommRing R✝
    x✝ : WittVector p R✝
    n : Nat
    ⊢ Eq (fun a => Matrix.vecCons x✝.coeff (Matrix.vecCons x✝.coeff Matrix.vecEmpt …
  -/
  ext ⟨i, k⟩
  /-
    case h.mk
    p : Nat
    R S : Type u
    idx : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    f : ⦃R : Type u_2⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).c …
    R✝ : Type u_2
    inst✝ : CommRing R✝
    x✝ : WittVector p R✝
    n : Nat
    i : Fin 2
    k : Nat
    ⊢ Eq (Matrix.vecCons x✝.coeff (Matrix.vecCons x✝.coeff Matrix.vecEmpty) { fst  …
  -/
                  /-
                    🎉 no goals
                  -/
  fin_cases i <;> simp
                  /-
                    🎉 no goals
                  -/

-- Porting note: Lean 4's typeclass inference is sufficiently more powerful that we no longer
-- need the `@[is_poly]` attribute. Use of the attribute should just be replaced by changing the
-- theorem to an `instance`.


/-- The additive negation is a polynomial function on Witt vectors. -/
-- Porting note: replaced `@[is_poly]` with `instance`.
instance negIsPoly [Fact p.Prime] : IsPoly p fun R _ => @Neg.neg (𝕎 R) _ :=
  ⟨⟨fun n => rename Prod.snd (wittNeg p n), by
      /-
        p : Nat
        R S : Type u
        idx : Type u_1
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Fact (Nat.Prime p)
        ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (Neg.neg x).co …
      -/
      intros; funext n
      /-
        case h
        p : Nat
        R S : Type u
        idx : Type u_1
        inst✝³ : CommRing R
        inst✝² : CommRing S
        inst✝¹ : Fact (Nat.Prime p)
        R✝ : Type u_2
        inst✝ : CommRing R✝
        x✝ : WittVector p R✝
        n : Nat
        ⊢ Eq ((Neg.neg x✝).coeff n) ((MvPolynomial.aeval x✝.coeff) ((fun n => (MvPolyn …
      -/
      rw [neg_coeff, aeval_eq_eval₂Hom, eval₂Hom_rename]
      /-
        case h
        p : Nat
        R S : Type u
        idx : Type u_1
        inst✝³ : CommRing R
        inst✝² : CommRing S
        inst✝¹ : Fact (Nat.Prime p)
        R✝ : Type u_2
        inst✝ : CommRing R✝
        x✝ : WittVector p R✝
        n : Nat
        ⊢ Eq (WittVector.peval (WittVector.wittNeg p n) (Matrix.vecCons x✝.coeff Matri …
      -/
      apply eval₂Hom_congr rfl _ rfl
      /-
        p : Nat
        R S : Type u
        idx : Type u_1
        inst✝³ : CommRing R
        inst✝² : CommRing S
        inst✝¹ : Fact (Nat.Prime p)
        R✝ : Type u_2
        inst✝ : CommRing R✝
        x✝ : WittVector p R✝
        n : Nat
        ⊢ Eq (Function.uncurry (Matrix.vecCons x✝.coeff Matrix.vecEmpty)) (Function.co …
      -/
      ext ⟨i, k⟩; fin_cases i; rfl⟩⟩
                               /-
                                 🎉 no goals
                               -/


/-- The function that is constantly zero on Witt vectors is a polynomial function. -/
instance zeroIsPoly [Fact p.Prime] : IsPoly p fun _ _ _ => 0 :=
          /-
            p : Nat
            R S : Type u
            idx : Type u_1
            inst✝² : CommRing R
            inst✝¹ : CommRing S
            inst✝ : Fact (Nat.Prime p)
            ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (WittVector.co …
          -/
  ⟨⟨0, by intros; funext n; simp only [Pi.zero_apply, map_zero, zero_coeff]⟩⟩
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem bind₁_zero_wittPolynomial [Fact p.Prime] (n : ℕ) :
    bind₁ (0 : ℕ → MvPolynomial ℕ R) (wittPolynomial p R n) = 0 := by
  /-
    p : Nat
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((MvPolynomial.bind₁ 0) (wittPolynomial p R n)) 0
  -/
  rw [← aeval_eq_bind₁, aeval_zero, constantCoeff_wittPolynomial, RingHom.map_zero]
  /-
    🎉 no goals
  -/


/-- The coefficients of `1 : 𝕎 R` as polynomials. -/
def onePoly (n : ℕ) : MvPolynomial ℕ ℤ :=
  if n = 0 then 1 else 0


@[simp]
theorem bind₁_onePoly_wittPolynomial [hp : Fact p.Prime] (n : ℕ) :
    bind₁ onePoly (wittPolynomial p ℤ n) = 1 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((MvPolynomial.bind₁ WittVector.onePoly) (wittPolynomial p Int n)) 1
  -/
  rw [wittPolynomial_eq_sum_C_mul_X_pow, map_sum, Finset.sum_eq_single 0]
  · simp only [onePoly, one_pow, one_mul, map_pow, C_1, pow_zero, bind₁_X_right, if_true,
      eq_self_iff_true]
    /-
      case h₀
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ ∀ (b : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) b → Ne b 0 → Eq ( …
    -/
  · intro i _hi hi0
    simp only [onePoly, if_neg hi0, zero_pow (pow_ne_zero _ hp.1.ne_zero), mul_zero, map_pow,
      bind₁_X_right, map_mul]
    /-
      case h₁
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ Not (Membership.mem (Finset.range (HAdd.hAdd n 1)) 0) → Eq ((MvPolynomial.bi …
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- The function that is constantly one on Witt vectors is a polynomial function. -/
instance oneIsPoly [Fact p.Prime] : IsPoly p fun _ _ _ => 1 :=
  ⟨⟨onePoly, by
      /-
        p : Nat
        R S : Type u
        idx : Type u_1
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Fact (Nat.Prime p)
        ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (WittVector.co …
      -/
      intros; funext n; cases n
        /-
          case h.zero
          p : Nat
          R S : Type u
          idx : Type u_1
          inst✝³ : CommRing R
          inst✝² : CommRing S
          inst✝¹ : Fact (Nat.Prime p)
          R✝ : Type u_2
          inst✝ : CommRing R✝
          x✝ : WittVector p R✝
          ⊢ Eq (WittVector.coeff 1 0) ((MvPolynomial.aeval x✝.coeff) (WittVector.onePoly …
        -/
      · simp only [lt_self_iff_false, one_coeff_zero, onePoly, ite_true, map_one]
        /-
          🎉 no goals
        -/
      · simp only [Nat.succ_pos', one_coeff_eq_of_pos, onePoly, Nat.succ_ne_zero, ite_false,
          map_zero]
  ⟩⟩


/-- Addition of Witt vectors is a polynomial function. -/
-- Porting note: replaced `@[is_poly]` with `instance`.
instance addIsPoly₂ [Fact p.Prime] : IsPoly₂ p fun _ _ => (· + ·) :=
                  /-
                    p : Nat
                    R S : Type u
                    idx : Type u_1
                    inst✝² : CommRing R
                    inst✝¹ : CommRing S
                    inst✝ : Fact (Nat.Prime p)
                    ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (HAdd.hAdd x …
                  -/
  ⟨⟨wittAdd p, by intros; ext; exact add_coeff _ _ _⟩⟩
                               /-
                                 🎉 no goals
                               -/


/-- Multiplication of Witt vectors is a polynomial function. -/
-- Porting note: replaced `@[is_poly]` with `instance`.
instance mulIsPoly₂ [Fact p.Prime] : IsPoly₂ p fun _ _ => (· * ·) :=
                  /-
                    p : Nat
                    R S : Type u
                    idx : Type u_1
                    inst✝² : CommRing R
                    inst✝¹ : CommRing S
                    inst✝ : Fact (Nat.Prime p)
                    ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x y : WittVector p R), Eq (HMul.hMul x …
                  -/
  ⟨⟨wittMul p, by intros; ext; exact mul_coeff _ _ _⟩⟩
                               /-
                                 🎉 no goals
                               -/

-- unfortunately this is not universe polymorphic, merely because `f` isn't

theorem IsPoly.map [Fact p.Prime] {f} (hf : IsPoly p f) (g : R →+* S) (x : 𝕎 R) :
    map g (f x) = f (map g x) := by
  -- this could be turned into a tactic “macro” (taking `hf` as parameter)
  -- so that applications do not have to worry about the universe issue
  -- see `IsPoly₂.map` for a slightly more general proof strategy
  /-
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    hf : WittVector.IsPoly p f
    g : RingHom R S
    x : WittVector p R
    ⊢ Eq ((WittVector.map g) (f x)) (f ((WittVector.map g) x))
  -/
  obtain ⟨φ, hf⟩ := hf
  /-
    case mk'.intro
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    g : RingHom R S
    x : WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    ⊢ Eq ((WittVector.map g) (f x)) (f ((WittVector.map g) x))
  -/
  ext n
  /-
    case mk'.intro.h
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    g : RingHom R S
    x : WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    n : Nat
    ⊢ Eq (((WittVector.map g) (f x)).coeff n) ((f ((WittVector.map g) x)).coeff n)
  -/
  simp only [map_coeff, hf, map_aeval]
  /-
    case mk'.intro.h
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    g : RingHom R S
    x : WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    n : Nat
    ⊢ Eq ((MvPolynomial.eval₂Hom (g.comp (algebraMap Int R)) fun i => g (x.coeff i …
  -/
  apply eval₂Hom_congr (RingHom.ext_int _ _) _ rfl
  /-
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    g : RingHom R S
    x : WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    n : Nat
    ⊢ Eq (fun i => g (x.coeff i)) ((WittVector.map g) x).coeff
  -/
  ext  -- Porting note: this `ext` was not present in the mathport output
  /-
    case h
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R
    g : RingHom R S
    x : WittVector p R
    φ : Nat → MvPolynomial Nat Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x : WittVector p R), Eq (f x).coeff f …
    n x✝ : Nat
    ⊢ Eq (g (x.coeff x✝)) (((WittVector.map g) x).coeff x✝)
  -/
  simp only [map_coeff]
  /-
    🎉 no goals
  -/


instance [Fact p.Prime] : Inhabited (IsPoly₂ p (fun _ _ => (· + ·))) :=
  ⟨addIsPoly₂⟩


theorem ext [Fact p.Prime] {f g} (hf : IsPoly₂ p f) (hg : IsPoly₂ p g)
    (h : ∀ (R : Type u) [_Rcr : CommRing R] (x y : 𝕎 R) (n : ℕ),
        ghostComponent n (f x y) = ghostComponent n (g x y)) :
    ∀ (R) [_Rcr : CommRing R] (x y : 𝕎 R), f x y = g x y := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    hf : WittVector.IsPoly₂ p f
    hg : WittVector.IsPoly₂ p g
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R) (n : Nat), Eq (( …
    ⊢ ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R), Eq (f x y) (g x y)
  -/
  obtain ⟨φ, hf⟩ := hf
  /-
    case mk'.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    hg : WittVector.IsPoly₂ p g
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R) (n : Nat), Eq (( …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ⊢ ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R), Eq (f x y) (g x y)
  -/
  obtain ⟨ψ, hg⟩ := hg
  /-
    case mk'.intro.mk'.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R) (n : Nat), Eq (( …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    ⊢ ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R), Eq (f x y) (g x y)
  -/
  intros
  /-
    case mk'.intro.mk'.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R) (n : Nat), Eq (( …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    ⊢ Eq (f x✝ y✝) (g x✝ y✝)
  -/
  ext n
  /-
    case mk'.intro.mk'.intro.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R) (n : Nat), Eq (( …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n : Nat
    ⊢ Eq ((f x✝ y✝).coeff n) ((g x✝ y✝).coeff n)
  -/
  rw [hf, hg, poly_eq_of_wittPolynomial_bind_eq' p φ ψ]
  /-
    case mk'.intro.mk'.intro.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R) (n : Nat), Eq (( …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n : Nat
    ⊢ ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPolyno …
  -/
  intro k
  /-
    case mk'.intro.mk'.intro.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R) (n : Nat), Eq (( …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n k : Nat
    ⊢ Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int k)) ((MvPolynomial.bind₁ ψ) …
  -/
  apply MvPolynomial.funext
  /-
    case mk'.intro.mk'.intro.h.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R) (n : Nat), Eq (( …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n k : Nat
    ⊢ ∀ (x : Prod (Fin 2) Nat → Int), Eq ((MvPolynomial.eval x) ((MvPolynomial.bin …
  -/
  intro x
  /-
    case mk'.intro.mk'.intro.h.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R) (n : Nat), Eq (( …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n k : Nat
    x : Prod (Fin 2) Nat → Int
    ⊢ Eq ((MvPolynomial.eval x) ((MvPolynomial.bind₁ φ) (wittPolynomial p Int k))) …
  -/
  simp only [hom_bind₁]
  /-
    case mk'.intro.mk'.intro.h.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    h : ∀ (R : Type u) [_Rcr : CommRing R] (x y : WittVector p R) (n : Nat), Eq (( …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n k : Nat
    x : Prod (Fin 2) Nat → Int
    ⊢ Eq ((MvPolynomial.eval₂Hom ((MvPolynomial.eval x).comp MvPolynomial.C) fun i …
  -/
  specialize h (ULift ℤ) (mk p fun i => ⟨x (0, i)⟩) (mk p fun i => ⟨x (1, i)⟩) k
  /-
    case mk'.intro.mk'.intro.h.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n k : Nat
    x : Prod (Fin 2) Nat → Int
    h : Eq ((WittVector.ghostComponent k) (f (WittVector.mk p fun i => { down := x …
    ⊢ Eq ((MvPolynomial.eval₂Hom ((MvPolynomial.eval x).comp MvPolynomial.C) fun i …
  -/
  simp only [ghostComponent_apply, aeval_eq_eval₂Hom] at h
  /-
    case mk'.intro.mk'.intro.h.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n k : Nat
    x : Prod (Fin 2) Nat → Int
    h : Eq ((MvPolynomial.eval₂Hom (algebraMap Int (ULift.{u, 0} Int)) (f (WittVec …
    ⊢ Eq ((MvPolynomial.eval₂Hom ((MvPolynomial.eval x).comp MvPolynomial.C) fun i …
  -/
  apply (ULift.ringEquiv.symm : ℤ ≃+* _).injective
  /-
    case mk'.intro.mk'.intro.h.h.a
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n k : Nat
    x : Prod (Fin 2) Nat → Int
    h : Eq ((MvPolynomial.eval₂Hom (algebraMap Int (ULift.{u, 0} Int)) (f (WittVec …
    ⊢ Eq (ULift.ringEquiv.symm ((MvPolynomial.eval₂Hom ((MvPolynomial.eval x).comp …
  -/
  simp only [← RingEquiv.coe_toRingHom, map_eval₂Hom]
  /-
    case mk'.intro.mk'.intro.h.h.a
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    f g : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → W …
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ψ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hg : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (g x y).coe …
    R✝ : Type u
    _Rcr✝ : CommRing R✝
    x✝ y✝ : WittVector p R✝
    n k : Nat
    x : Prod (Fin 2) Nat → Int
    h : Eq ((MvPolynomial.eval₂Hom (algebraMap Int (ULift.{u, 0} Int)) (f (WittVec …
    ⊢ Eq ((MvPolynomial.eval₂Hom ((↑ULift.ringEquiv.symm).comp ((MvPolynomial.eval …
  -/
  convert h using 1
  all_goals
    simp only [hf, hg, MvPolynomial.eval, map_eval₂Hom]
    apply eval₂Hom_congr (RingHom.ext_int _ _) _ rfl
    ext1
    apply eval₂Hom_congr (RingHom.ext_int _ _) _ rfl
    ext ⟨b, _⟩
    fin_cases b <;> simp only [coeff_mk, uncurry] <;> rfl

-- unfortunately this is not universe polymorphic, merely because `f` isn't

theorem map [Fact p.Prime] {f} (hf : IsPoly₂ p f) (g : R →+* S) (x y : 𝕎 R) :
    map g (f x y) = f (map g x) (map g y) := by
  -- this could be turned into a tactic “macro” (taking `hf` as parameter)
  -- so that applications do not have to worry about the universe issue
  /-
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → Wit …
    hf : WittVector.IsPoly₂ p f
    g : RingHom R S
    x y : WittVector p R
    ⊢ Eq ((WittVector.map g) (f x y)) (f ((WittVector.map g) x) ((WittVector.map g …
  -/
  obtain ⟨φ, hf⟩ := hf
  /-
    case mk'.intro
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → Wit …
    g : RingHom R S
    x y : WittVector p R
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    ⊢ Eq ((WittVector.map g) (f x y)) (f ((WittVector.map g) x) ((WittVector.map g …
  -/
  ext n
  /-
    case mk'.intro.h
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → Wit …
    g : RingHom R S
    x y : WittVector p R
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    n : Nat
    ⊢ Eq (((WittVector.map g) (f x y)).coeff n) ((f ((WittVector.map g) x) ((WittV …
  -/
  simp (config := { unfoldPartialApp := true }) only [map_coeff, hf, map_aeval, peval, uncurry]
  /-
    case mk'.intro.h
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → Wit …
    g : RingHom R S
    x y : WittVector p R
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    n : Nat
    ⊢ Eq ((MvPolynomial.eval₂Hom (g.comp (algebraMap Int R)) fun i => g (Matrix.ve …
  -/
  apply eval₂Hom_congr (RingHom.ext_int _ _) _ rfl
  /-
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → Wit …
    g : RingHom R S
    x y : WittVector p R
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    n : Nat
    ⊢ Eq (fun i => g (Matrix.vecCons x.coeff (Matrix.vecCons y.coeff Matrix.vecEmp …
  -/
  ext ⟨i, k⟩
  /-
    case h.mk
    p : Nat
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Fact (Nat.Prime p)
    f : ⦃R : Type u⦄ → [inst : CommRing R] → WittVector p R → WittVector p R → Wit …
    g : RingHom R S
    x y : WittVector p R
    φ : Nat → MvPolynomial (Prod (Fin 2) Nat) Int
    hf : ∀ ⦃R : Type u⦄ [inst : CommRing R] (x y : WittVector p R), Eq (f x y).coe …
    n : Nat
    i : Fin 2
    k : Nat
    ⊢ Eq (g (Matrix.vecCons x.coeff (Matrix.vecCons y.coeff Matrix.vecEmpty) { fst …
  -/
                  /-
                    🎉 no goals
                  -/
  fin_cases i <;> simp
                  /-
                    🎉 no goals
                  -/


attribute [ghost_simps] AlgHom.id_apply map_natCast RingHom.map_zero RingHom.map_one RingHom.map_mul
  RingHom.map_add RingHom.map_sub RingHom.map_neg RingHom.id_apply mul_add add_mul add_zero zero_add
  mul_one one_mul mul_zero zero_mul Nat.succ_ne_zero add_tsub_cancel_right
  Nat.succ_eq_add_one if_true eq_self_iff_true if_false forall_true_iff forall₂_true_iff
  forall₃_true_iff


/-- A macro for a common simplification when rewriting with ghost component equations. -/
syntax (name := ghostSimp) "ghost_simp" (simpArgs)? : tactic


macro_rules
  | `(tactic| ghost_simp $[[$simpArgs,*]]?) => do
    let args := simpArgs.map (·.getElems) |>.getD #[]
    `(tactic| simp only [← sub_eq_add_neg, ghost_simps, $args,*])



/-- `ghost_calc` is a tactic for proving identities between polynomial functions.
Typically, when faced with a goal like
```lean
∀ (x y : 𝕎 R), verschiebung (x * frobenius y) = verschiebung x * y
```
you can
1. call `ghost_calc`
2. do a small amount of manual work -- maybe nothing, maybe `rintro`, etc
3. call `ghost_simp`

and this will close the goal.

`ghost_calc` cannot detect whether you are dealing with unary or binary polynomial functions.
You must give it arguments to determine this.
If you are proving a universally quantified goal like the above,
call `ghost_calc _ _`.
If the variables are introduced already, call `ghost_calc x y`.
In the unary case, use `ghost_calc _` or `ghost_calc x`.

`ghost_calc` is a light wrapper around type class inference.
All it does is apply the appropriate extensionality lemma and try to infer the resulting goals.
This is subtle and Lean's elaborator doesn't like it because of the HO unification involved,
so it is easier (and prettier) to put it in a tactic script.
-/
syntax (name := ghostCalc) "ghost_calc" (ppSpace colGt term:max)* : tactic


private def runIntro (ref : Syntax) (n : Name) : TacticM FVarId := do
  let fvarId ← liftMetaTacticAux fun g => do
    let (fv, g') ← g.intro n
    return (fv, [g'])
  withMainContext do
    Elab.Term.addLocalVarInfo ref (mkFVar fvarId)
  return fvarId


private def getLocalOrIntro (t : Term) : TacticM FVarId := do
  match t with
    | `(_) => runIntro t `_
    | `($id:ident) => getFVarId id <|> runIntro id id.getId
    | _ => Elab.throwUnsupportedSyntax


elab_rules : tactic | `(tactic| ghost_calc $[$ids']*) => do
  let ids ← ids'.mapM getLocalOrIntro
  withMainContext do
  let idsS ← ids.mapM (fun id => Elab.Term.exprToSyntax (.fvar id))
  let some (α, lhs, rhs) := (← getMainTarget'').eq?
    | throwError "ghost_calc expecting target to be an equality"
  let (``WittVector, #[_, R]) := α.getAppFnArgs
    | throwError "ghost_calc expecting target to be an equality of `WittVector`s"
  let instR ← Meta.synthInstance (← Meta.mkAppM ``CommRing #[R])
  unless instR.isFVar do
    throwError "{← Meta.inferType instR} instance is not local"
  let f ← Meta.mkLambdaFVars (#[R, instR] ++ ids.map .fvar) lhs
  let g ← Meta.mkLambdaFVars (#[R, instR] ++ ids.map .fvar) rhs
  let fS ← Elab.Term.exprToSyntax f
  let gS ← Elab.Term.exprToSyntax g
  match idsS with
    | #[x] => evalTactic (← `(tactic| refine IsPoly.ext (f := $fS) (g := $gS) ?_ ?_ ?_ _ $x))
    | #[x, y] => evalTactic (← `(tactic| refine IsPoly₂.ext (f := $fS) (g := $gS) ?_ ?_ ?_ _ $x $y))
    | _ => throwError "ghost_calc takes either one or two arguments"
  let nm ← withMainContext <|
    if let .fvar fvarId := (R : Expr) then
      fvarId.getUserName
    else
      Meta.getUnusedUserName `R
  evalTactic <| ← `(tactic| iterate 2 infer_instance)
  let R := mkIdent nm
  evalTactic <| ← `(tactic| clear! $R)
  evalTactic <| ← `(tactic| intro $(mkIdent nm):ident $(mkIdent (.str nm "_inst")):ident $ids'*)


