/-- A ring is a Jacobson ring if for every radical ideal `I`,
 the Jacobson radical of `I` is equal to `I`.
 See `isJacobsonRing_iff_prime_eq` and `isJacobsonRing_iff_sInf_maximal`
 for equivalent definitions. -/
class IsJacobsonRing (R : Type*) [CommRing R] : Prop where
  out' : ∀ I : Ideal R, I.IsRadical → I.jacobson = I


theorem isJacobsonRing_iff {R} [CommRing R] :
    IsJacobsonRing R ↔ ∀ I : Ideal R, I.IsRadical → I.jacobson = I :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


theorem IsJacobsonRing.out {R} [CommRing R] :
    IsJacobsonRing R → ∀ {I : Ideal R}, I.IsRadical → I.jacobson = I :=
  isJacobsonRing_iff.1


/-- A ring is a Jacobson ring if and only if for all prime ideals `P`,
 the Jacobson radical of `P` is equal to `P`. -/
theorem isJacobsonRing_iff_prime_eq :
    IsJacobsonRing R ↔ ∀ P : Ideal R, IsPrime P → P.jacobson = P := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Iff (IsJacobsonRing R) (∀ (P : Ideal R), P.IsPrime → Eq P.jacobson P)
  -/
  refine isJacobsonRing_iff.trans ⟨fun h I hI => h I hI.isRadical, ?_⟩
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ (∀ (P : Ideal R), P.IsPrime → Eq P.jacobson P) → ∀ (I : Ideal R), I.IsRadica …
  -/
  refine fun h I hI ↦ le_antisymm (fun x hx ↦ ?_) (fun x hx ↦ mem_sInf.mpr fun _ hJ ↦ hJ.left hx)
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (P : Ideal R), P.IsPrime → Eq P.jacobson P
    I : Ideal R
    hI : I.IsRadical
    x : R
    hx : Membership.mem I.jacobson x
    ⊢ Membership.mem I x
  -/
  rw [← hI.radical, radical_eq_sInf I, mem_sInf]
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (P : Ideal R), P.IsPrime → Eq P.jacobson P
    I : Ideal R
    hI : I.IsRadical
    x : R
    hx : Membership.mem I.jacobson x
    ⊢ ∀ ⦃I_1 : Ideal R⦄, Membership.mem (setOf fun J => And (LE.le I J) J.IsPrime) …
  -/
  intro P hP
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (P : Ideal R), P.IsPrime → Eq P.jacobson P
    I : Ideal R
    hI : I.IsRadical
    x : R
    hx : Membership.mem I.jacobson x
    P : Ideal R
    hP : Membership.mem (setOf fun J => And (LE.le I J) J.IsPrime) P
    ⊢ Membership.mem P x
  -/
  rw [Set.mem_setOf_eq] at hP
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (P : Ideal R), P.IsPrime → Eq P.jacobson P
    I : Ideal R
    hI : I.IsRadical
    x : R
    hx : Membership.mem I.jacobson x
    P : Ideal R
    hP : And (LE.le I P) P.IsPrime
    ⊢ Membership.mem P x
  -/
  erw [mem_sInf] at hx
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (P : Ideal R), P.IsPrime → Eq P.jacobson P
    I : Ideal R
    hI : I.IsRadical
    x : R
    hx : ∀ ⦃I_1 : Ideal R⦄, Membership.mem (setOf fun J => And (LE.le I J) J.IsMax …
    P : Ideal R
    hP : And (LE.le I P) P.IsPrime
    ⊢ Membership.mem P x
  -/
  erw [← h P hP.right, mem_sInf]
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (P : Ideal R), P.IsPrime → Eq P.jacobson P
    I : Ideal R
    hI : I.IsRadical
    x : R
    hx : ∀ ⦃I_1 : Ideal R⦄, Membership.mem (setOf fun J => And (LE.le I J) J.IsMax …
    P : Ideal R
    hP : And (LE.le I P) P.IsPrime
    ⊢ ∀ ⦃I : Ideal R⦄, Membership.mem (setOf fun J => And (LE.le P J) J.IsMaximal) …
  -/
  exact fun J hJ => hx ⟨le_trans hP.left hJ.left, hJ.right⟩
  /-
    🎉 no goals
  -/


/-- A ring `R` is Jacobson if and only if for every prime ideal `I`,
 `I` can be written as the infimum of some collection of maximal ideals.
 Allowing ⊤ in the set `M` of maximal ideals is equivalent, but makes some proofs cleaner. -/
theorem isJacobsonRing_iff_sInf_maximal : IsJacobsonRing R ↔ ∀ {I : Ideal R}, I.IsPrime →
    ∃ M : Set (Ideal R), (∀ J ∈ M, IsMaximal J ∨ J = ⊤) ∧ I = sInf M :=
  ⟨fun H _I h => eq_jacobson_iff_sInf_maximal.1 (H.out h.isRadical), fun H =>
    isJacobsonRing_iff_prime_eq.2 fun _P hP => eq_jacobson_iff_sInf_maximal.2 (H hP)⟩


/-- A variant of `isJacobsonRing_iff_sInf_maximal` with a different spelling of "maximal or `⊤`". -/
theorem isJacobsonRing_iff_sInf_maximal' : IsJacobsonRing R ↔ ∀ {I : Ideal R}, I.IsPrime →
    ∃ M : Set (Ideal R), (∀ J ∈ M, ∀ (K : Ideal R), J < K → K = ⊤) ∧ I = sInf M :=
  ⟨fun H _I h => eq_jacobson_iff_sInf_maximal'.1 (H.out h.isRadical), fun H =>
    isJacobsonRing_iff_prime_eq.2 fun _P hP => eq_jacobson_iff_sInf_maximal'.2 (H hP)⟩


theorem Ideal.radical_eq_jacobson [H : IsJacobsonRing R] (I : Ideal R) : I.radical = I.jacobson :=
  le_antisymm (le_sInf fun _J ⟨hJ, hJ_max⟩ => (IsPrime.radical_le_iff hJ_max.isPrime).mpr hJ)
    (H.out (radical_isRadical I) ▸ jacobson_mono le_radical)


instance (priority := 100) [IsArtinianRing R] : IsJacobsonRing R :=
  isJacobsonRing_iff_prime_eq.mpr fun P _ ↦
    jacobson_eq_self_of_isMaximal (H := IsArtinianRing.isMaximal_of_isPrime P)


theorem isJacobsonRing_of_surjective [H : IsJacobsonRing R] :
    (∃ f : R →+* S, Function.Surjective ↑f) → IsJacobsonRing S := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    H : IsJacobsonRing R
    ⊢ (Exists fun f => Function.Surjective ⇑f) → IsJacobsonRing S
  -/
  rintro ⟨f, hf⟩
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    H : IsJacobsonRing R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    ⊢ IsJacobsonRing S
  -/
  rw [isJacobsonRing_iff_sInf_maximal]
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    H : IsJacobsonRing R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    ⊢ ∀ {I : Ideal S}, I.IsPrime → Exists fun M => And (∀ (J : Ideal S), Membershi …
  -/
  intro p hp
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    H : IsJacobsonRing R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    p : Ideal S
    hp : p.IsPrime
    ⊢ Exists fun M => And (∀ (J : Ideal S), Membership.mem M J → Or J.IsMaximal (E …
  -/
  use map f '' { J : Ideal R | comap f p ≤ J ∧ J.IsMaximal }
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    H : IsJacobsonRing R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    p : Ideal S
    hp : p.IsPrime
    ⊢ And (∀ (J : Ideal S), Membership.mem (Set.image (Ideal.map f) (setOf fun J = …
  -/
  use fun j ⟨J, hJ, hmap⟩ => hmap ▸ (map_eq_top_or_isMaximal_of_surjective f hf hJ.right).symm
  have : p = map f (comap f p).jacobson :=
    (IsJacobsonRing.out' _ <| hp.isRadical.comap f).symm ▸ (map_comap_of_surjective f hf p).symm
  /-
    case right
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    H : IsJacobsonRing R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    p : Ideal S
    hp : p.IsPrime
    this : Eq p (Ideal.map f (Ideal.comap f p).jacobson)
    ⊢ Eq p (InfSet.sInf (Set.image (Ideal.map f) (setOf fun J => And (LE.le (Ideal …
  -/
  exact this.trans (map_sInf hf fun J ⟨hJ, _⟩ => le_trans (Ideal.ker_le_comap f) hJ)
  /-
    🎉 no goals
  -/


instance (priority := 100) isJacobsonRing_quotient [IsJacobsonRing R] : IsJacobsonRing (R ⧸ I) :=
  isJacobsonRing_of_surjective ⟨Ideal.Quotient.mk I, by
    /-
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      I : Ideal R
      inst✝ : IsJacobsonRing R
      ⊢ Function.Surjective ⇑(Ideal.Quotient.mk I)
    -/
    rintro ⟨x⟩
    /-
      case mk
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      I : Ideal R
      inst✝ : IsJacobsonRing R
      b✝ : HasQuotient.Quotient R I
      x : R
      ⊢ Exists fun a => Eq ((Ideal.Quotient.mk I) a) (Quot.mk (⇑(Submodule.quotientR …
    -/
    use x
    /-
      case h
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      I : Ideal R
      inst✝ : IsJacobsonRing R
      b✝ : HasQuotient.Quotient R I
      x : R
      ⊢ Eq ((Ideal.Quotient.mk I) x) (Quot.mk (⇑(Submodule.quotientRel I)) x)
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


theorem isJacobsonRing_iso (e : R ≃+* S) : IsJacobsonRing R ↔ IsJacobsonRing S :=
  ⟨fun h => @isJacobsonRing_of_surjective _ _ _ _ h ⟨(e : R →+* S), e.surjective⟩, fun h =>
    @isJacobsonRing_of_surjective _ _ _ _ h ⟨(e.symm : S →+* R), e.symm.surjective⟩⟩


theorem isJacobsonRing_of_isIntegral [Algebra R S] [Algebra.IsIntegral R S] [IsJacobsonRing R] :
    IsJacobsonRing S := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : Algebra.IsIntegral R S
    inst✝ : IsJacobsonRing R
    ⊢ IsJacobsonRing S
  -/
  rw [isJacobsonRing_iff_prime_eq]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : Algebra.IsIntegral R S
    inst✝ : IsJacobsonRing R
    ⊢ ∀ (P : Ideal S), P.IsPrime → Eq P.jacobson P
  -/
  intro P hP
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : Algebra.IsIntegral R S
    inst✝ : IsJacobsonRing R
    P : Ideal S
    hP : P.IsPrime
    ⊢ Eq P.jacobson P
  -/
  by_cases hP_top : comap (algebraMap R S) P = ⊤
    /-
      case pos
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : Algebra.IsIntegral R S
      inst✝ : IsJacobsonRing R
      P : Ideal S
      hP : P.IsPrime
      hP_top : Eq (Ideal.comap (algebraMap R S) P) Top.top
      ⊢ Eq P.jacobson P
    -/
  · simp [comap_eq_top_iff.1 hP_top]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : Algebra.IsIntegral R S
      inst✝ : IsJacobsonRing R
      P : Ideal S
      hP : P.IsPrime
      hP_top : Not (Eq (Ideal.comap (algebraMap R S) P) Top.top)
      ⊢ Eq P.jacobson P
    -/
  · haveI : Nontrivial (R ⧸ comap (algebraMap R S) P) := Quotient.nontrivial hP_top
    /-
      case neg
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : Algebra.IsIntegral R S
      inst✝ : IsJacobsonRing R
      P : Ideal S
      hP : P.IsPrime
      hP_top : Not (Eq (Ideal.comap (algebraMap R S) P) Top.top)
      this : Nontrivial (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) P))
      ⊢ Eq P.jacobson P
    -/
    rw [jacobson_eq_iff_jacobson_quotient_eq_bot]
    /-
      case neg
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : Algebra.IsIntegral R S
      inst✝ : IsJacobsonRing R
      P : Ideal S
      hP : P.IsPrime
      hP_top : Not (Eq (Ideal.comap (algebraMap R S) P) Top.top)
      this : Nontrivial (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) P))
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    refine eq_bot_of_comap_eq_bot (R := R ⧸ comap (algebraMap R S) P) ?_
    rw [eq_bot_iff, ← jacobson_eq_iff_jacobson_quotient_eq_bot.1
      ((isJacobsonRing_iff_prime_eq.1 ‹_›) (comap (algebraMap R S) P) (comap_isPrime _ _)),
      comap_jacobson]
    /-
      case neg
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : Algebra.IsIntegral R S
      inst✝ : IsJacobsonRing R
      P : Ideal S
      hP : P.IsPrime
      hP_top : Not (Eq (Ideal.comap (algebraMap R S) P) Top.top)
      this : Nontrivial (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) P))
      ⊢ LE.le (InfSet.sInf (Set.image (Ideal.comap (algebraMap (HasQuotient.Quotient …
    -/
    refine sInf_le_sInf fun J hJ => ?_
    /-
      case neg
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : Algebra.IsIntegral R S
      inst✝ : IsJacobsonRing R
      P : Ideal S
      hP : P.IsPrime
      hP_top : Not (Eq (Ideal.comap (algebraMap R S) P) Top.top)
      this : Nontrivial (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) P))
      J : Ideal (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) P))
      hJ : Membership.mem (setOf fun J => And (LE.le Bot.bot J) J.IsMaximal) J
      ⊢ Membership.mem (Set.image (Ideal.comap (algebraMap (HasQuotient.Quotient R ( …
    -/
    simp only [true_and, Set.mem_image, bot_le, Set.mem_setOf_eq]
    /-
      case neg
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : Algebra.IsIntegral R S
      inst✝ : IsJacobsonRing R
      P : Ideal S
      hP : P.IsPrime
      hP_top : Not (Eq (Ideal.comap (algebraMap R S) P) Top.top)
      this : Nontrivial (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) P))
      J : Ideal (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) P))
      hJ : Membership.mem (setOf fun J => And (LE.le Bot.bot J) J.IsMaximal) J
      ⊢ Exists fun x => And x.IsMaximal (Eq (Ideal.comap (algebraMap (HasQuotient.Qu …
    -/
    have : J.IsMaximal := by simpa using hJ
    exact exists_ideal_over_maximal_of_isIntegral J
      (comap_bot_le_of_injective _ algebraMap_quotient_injective)


/-- A variant of `isJacobsonRing_of_isIntegral` that takes `RingHom.IsIntegral` instead. -/
theorem isJacobsonRing_of_isIntegral' (f : R →+* S) (hf : f.IsIntegral) [IsJacobsonRing R] :
    IsJacobsonRing S :=
  let _ : Algebra R S := f.toAlgebra
  have : Algebra.IsIntegral R S := ⟨hf⟩
  isJacobsonRing_of_isIntegral (R := R)


/-- If `R` is a Jacobson ring, then maximal ideals in the localization at `y`
correspond to maximal ideals in the original ring `R` that don't contain `y`.
This lemma gives the correspondence in the particular case of an ideal and its comap.
See `le_relIso_of_maximal` for the more general relation isomorphism -/
theorem IsLocalization.isMaximal_iff_isMaximal_disjoint [H : IsJacobsonRing R] (J : Ideal S) :
    J.IsMaximal ↔ (comap (algebraMap R S) J).IsMaximal ∧ y ∉ Ideal.comap (algebraMap R S) J := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    y : R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization.Away y S
    H : IsJacobsonRing R
    J : Ideal S
    ⊢ Iff J.IsMaximal (And (Ideal.comap (algebraMap R S) J).IsMaximal (Not (Member …
  -/
  constructor
  · refine fun h => ⟨?_, fun hy =>
      h.ne_top (Ideal.eq_top_of_isUnit_mem _ hy (map_units _ ⟨y, Submonoid.mem_powers _⟩))⟩
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      y : R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization.Away y S
      H : IsJacobsonRing R
      J : Ideal S
      h : J.IsMaximal
      ⊢ (Ideal.comap (algebraMap R S) J).IsMaximal
    -/
    have hJ : J.IsPrime := IsMaximal.isPrime h
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      y : R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization.Away y S
      H : IsJacobsonRing R
      J : Ideal S
      h : J.IsMaximal
      hJ : J.IsPrime
      ⊢ (Ideal.comap (algebraMap R S) J).IsMaximal
    -/
    rw [isPrime_iff_isPrime_disjoint (Submonoid.powers y)] at hJ
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      y : R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization.Away y S
      H : IsJacobsonRing R
      J : Ideal S
      h : J.IsMaximal
      hJ : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑(Submonoid.powers …
      ⊢ (Ideal.comap (algebraMap R S) J).IsMaximal
    -/
    have : y ∉ (comap (algebraMap R S) J).1 := Set.disjoint_left.1 hJ.right (Submonoid.mem_powers _)
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      y : R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization.Away y S
      H : IsJacobsonRing R
      J : Ideal S
      h : J.IsMaximal
      hJ : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑(Submonoid.powers …
      this : Not (Membership.mem (Ideal.comap (algebraMap R S) J).toAddSubmonoid y)
      ⊢ (Ideal.comap (algebraMap R S) J).IsMaximal
    -/
    erw [← H.out hJ.left.isRadical, Ideal.mem_sInf] at this
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      y : R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization.Away y S
      H : IsJacobsonRing R
      J : Ideal S
      h : J.IsMaximal
      hJ : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑(Submonoid.powers …
      this : Not (∀ ⦃I : Ideal R⦄, Membership.mem (setOf fun J_1 => And (LE.le (Idea …
      ⊢ (Ideal.comap (algebraMap R S) J).IsMaximal
    -/
    push_neg at this
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      y : R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization.Away y S
      H : IsJacobsonRing R
      J : Ideal S
      h : J.IsMaximal
      hJ : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑(Submonoid.powers …
      this : Exists fun ⦃I⦄ => And (Membership.mem (setOf fun J_1 => And (LE.le (Ide …
      ⊢ (Ideal.comap (algebraMap R S) J).IsMaximal
    -/
    rcases this with ⟨I, hI, hI'⟩
    /-
      case mp.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      y : R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization.Away y S
      H : IsJacobsonRing R
      J : Ideal S
      h : J.IsMaximal
      hJ : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑(Submonoid.powers …
      I : Ideal R
      hI : Membership.mem (setOf fun J_1 => And (LE.le (Ideal.comap (algebraMap R S) …
      hI' : Not (Membership.mem I y)
      ⊢ (Ideal.comap (algebraMap R S) J).IsMaximal
    -/
    convert hI.right
    /-
      case h.e'_3.h
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      y : R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization.Away y S
      H : IsJacobsonRing R
      J : Ideal S
      h : J.IsMaximal
      hJ : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑(Submonoid.powers …
      I : Ideal R
      hI : Membership.mem (setOf fun J_1 => And (LE.le (Ideal.comap (algebraMap R S) …
      hI' : Not (Membership.mem I y)
      e_2✝ : Eq CommSemiring.toSemiring Ring.toSemiring
      ⊢ Eq (Ideal.comap (algebraMap R S) J) I
    -/
    by_cases hJ : J = I.map (algebraMap R S)
      /-
        case pos
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        y : R
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization.Away y S
        H : IsJacobsonRing R
        J : Ideal S
        h : J.IsMaximal
        hJ✝ : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑(Submonoid.power …
        I : Ideal R
        hI : Membership.mem (setOf fun J_1 => And (LE.le (Ideal.comap (algebraMap R S) …
        hI' : Not (Membership.mem I y)
        e_2✝ : Eq CommSemiring.toSemiring Ring.toSemiring
        hJ : Eq J (Ideal.map (algebraMap R S) I)
        ⊢ Eq (Ideal.comap (algebraMap R S) J) I
      -/
    · rw [hJ, comap_map_of_isPrime_disjoint (powers y) S I (IsMaximal.isPrime hI.right)]
      /-
        case pos
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        y : R
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization.Away y S
        H : IsJacobsonRing R
        J : Ideal S
        h : J.IsMaximal
        hJ✝ : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑(Submonoid.power …
        I : Ideal R
        hI : Membership.mem (setOf fun J_1 => And (LE.le (Ideal.comap (algebraMap R S) …
        hI' : Not (Membership.mem I y)
        e_2✝ : Eq CommSemiring.toSemiring Ring.toSemiring
        hJ : Eq J (Ideal.map (algebraMap R S) I)
        ⊢ Disjoint ↑(Submonoid.powers y) ↑I
      -/
      rwa [disjoint_powers_iff_not_mem y hI.right.isPrime.isRadical]
      /-
        🎉 no goals
      -/
    · have hI_p : (I.map (algebraMap R S)).IsPrime := by
        refine isPrime_of_isPrime_disjoint (powers y) _ I hI.right.isPrime ?_
        rwa [disjoint_powers_iff_not_mem y hI.right.isPrime.isRadical]
      /-
        case neg
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        y : R
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization.Away y S
        H : IsJacobsonRing R
        J : Ideal S
        h : J.IsMaximal
        hJ✝ : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑(Submonoid.power …
        I : Ideal R
        hI : Membership.mem (setOf fun J_1 => And (LE.le (Ideal.comap (algebraMap R S) …
        hI' : Not (Membership.mem I y)
        e_2✝ : Eq CommSemiring.toSemiring Ring.toSemiring
        hJ : Not (Eq J (Ideal.map (algebraMap R S) I))
        hI_p : (Ideal.map (algebraMap R S) I).IsPrime
        ⊢ Eq (Ideal.comap (algebraMap R S) J) I
      -/
      have : J ≤ I.map (algebraMap R S) := map_comap (Submonoid.powers y) S J ▸ map_mono hI.left
      /-
        case neg
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        y : R
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization.Away y S
        H : IsJacobsonRing R
        J : Ideal S
        h : J.IsMaximal
        hJ✝ : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑(Submonoid.power …
        I : Ideal R
        hI : Membership.mem (setOf fun J_1 => And (LE.le (Ideal.comap (algebraMap R S) …
        hI' : Not (Membership.mem I y)
        e_2✝ : Eq CommSemiring.toSemiring Ring.toSemiring
        hJ : Not (Eq J (Ideal.map (algebraMap R S) I))
        hI_p : (Ideal.map (algebraMap R S) I).IsPrime
        this : LE.le J (Ideal.map (algebraMap R S) I)
        ⊢ Eq (Ideal.comap (algebraMap R S) J) I
      -/
      exact absurd (h.1.2 _ (lt_of_le_of_ne this hJ)) hI_p.1
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      y : R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization.Away y S
      H : IsJacobsonRing R
      J : Ideal S
      ⊢ And (Ideal.comap (algebraMap R S) J).IsMaximal (Not (Membership.mem (Ideal.c …
    -/
  · refine fun h => ⟨⟨fun hJ => h.1.ne_top (eq_top_iff.2 ?_), fun I hI => ?_⟩⟩
      /-
        case mpr.refine_1
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        y : R
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization.Away y S
        H : IsJacobsonRing R
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsMaximal (Not (Membership.mem (Ideal …
        hJ : Eq J Top.top
        ⊢ LE.le Top.top (Ideal.comap (algebraMap R S) J)
      -/
    · rwa [eq_top_iff, ← (IsLocalization.orderEmbedding (powers y) S).le_iff_le] at hJ
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_2
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        y : R
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization.Away y S
        H : IsJacobsonRing R
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsMaximal (Not (Membership.mem (Ideal …
        I : Ideal S
        hI : LT.lt J I
        ⊢ Eq I Top.top
      -/
    · have := congr_arg (Ideal.map (algebraMap R S)) (h.1.1.2 _ ⟨comap_mono (le_of_lt hI), ?_⟩)
        /-
          case mpr.refine_2.refine_2
          R : Type u_1
          S : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing S
          y : R
          inst✝¹ : Algebra R S
          inst✝ : IsLocalization.Away y S
          H : IsJacobsonRing R
          J : Ideal S
          h : And (Ideal.comap (algebraMap R S) J).IsMaximal (Not (Membership.mem (Ideal …
          I : Ideal S
          hI : LT.lt J I
          this : Eq (Ideal.map (algebraMap R S) (Ideal.comap (algebraMap R S) I)) (Ideal …
          ⊢ Eq I Top.top
        -/
      · rwa [map_comap (powers y) S I, Ideal.map_top] at this
        /-
          🎉 no goals
        -/
      /-
        case mpr.refine_2.refine_1
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        y : R
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization.Away y S
        H : IsJacobsonRing R
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsMaximal (Not (Membership.mem (Ideal …
        I : Ideal S
        hI : LT.lt J I
        ⊢ Not (HasSubset.Subset ↑(Ideal.comap (algebraMap R S) I) ↑(Ideal.comap (algeb …
      -/
      refine fun hI' => hI.right ?_
      /-
        case mpr.refine_2.refine_1
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        y : R
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization.Away y S
        H : IsJacobsonRing R
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsMaximal (Not (Membership.mem (Ideal …
        I : Ideal S
        hI : LT.lt J I
        hI' : HasSubset.Subset ↑(Ideal.comap (algebraMap R S) I) ↑(Ideal.comap (algebr …
        ⊢ HasSubset.Subset ↑I ↑J
      -/
      rw [← map_comap (powers y) S I, ← map_comap (powers y) S J]
      /-
        case mpr.refine_2.refine_1
        R : Type u_1
        S : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing S
        y : R
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization.Away y S
        H : IsJacobsonRing R
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsMaximal (Not (Membership.mem (Ideal …
        I : Ideal S
        hI : LT.lt J I
        hI' : HasSubset.Subset ↑(Ideal.comap (algebraMap R S) I) ↑(Ideal.comap (algebr …
        ⊢ HasSubset.Subset ↑(Ideal.map (algebraMap R S) (Ideal.comap (algebraMap R S)  …
      -/
      exact map_mono hI'
      /-
        🎉 no goals
      -/


/-- If `R` is a Jacobson ring, then maximal ideals in the localization at `y`
correspond to maximal ideals in the original ring `R` that don't contain `y`.
This lemma gives the correspondence in the particular case of an ideal and its map.
See `le_relIso_of_maximal` for the more general statement, and the reverse of this implication -/
theorem IsLocalization.isMaximal_of_isMaximal_disjoint
    [IsJacobsonRing R] (I : Ideal R) (hI : I.IsMaximal)
    (hy : y ∉ I) : (I.map (algebraMap R S)).IsMaximal := by
  rw [isMaximal_iff_isMaximal_disjoint S y,
    comap_map_of_isPrime_disjoint (powers y) S I (IsMaximal.isPrime hI)
      ((disjoint_powers_iff_not_mem y hI.isPrime.isRadical).2 hy)]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    y : R
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization.Away y S
    inst✝ : IsJacobsonRing R
    I : Ideal R
    hI : I.IsMaximal
    hy : Not (Membership.mem I y)
    ⊢ And I.IsMaximal (Not (Membership.mem I y))
  -/
  exact ⟨hI, hy⟩
  /-
    🎉 no goals
  -/


/-- If `R` is a Jacobson ring, then maximal ideals in the localization at `y`
correspond to maximal ideals in the original ring `R` that don't contain `y` -/
def IsLocalization.orderIsoOfMaximal [IsJacobsonRing R] :
    { p : Ideal S // p.IsMaximal } ≃o { p : Ideal R // p.IsMaximal ∧ y ∉ p } where
  toFun p := ⟨Ideal.comap (algebraMap R S) p.1, (isMaximal_iff_isMaximal_disjoint S y p.1).1 p.2⟩
  invFun p := ⟨Ideal.map (algebraMap R S) p.1, isMaximal_of_isMaximal_disjoint y p.1 p.2.1 p.2.2⟩
  left_inv J := Subtype.eq (map_comap (powers y) S J)
  right_inv I := Subtype.eq (comap_map_of_isPrime_disjoint _ _ I.1 (IsMaximal.isPrime I.2.1)
    ((disjoint_powers_iff_not_mem y I.2.1.isPrime.isRadical).2 I.2.2))
  map_rel_iff' {I I'} := ⟨fun h => show I.val ≤ I'.val from
    map_comap (powers y) S I.val ▸ map_comap (powers y) S I'.val ▸ Ideal.map_mono h,
    fun h _ hx => h hx⟩


include y in
/-- If `S` is the localization of the Jacobson ring `R` at the submonoid generated by `y : R`, then
`S` is Jacobson. -/
theorem isJacobsonRing_localization [H : IsJacobsonRing R] : IsJacobsonRing S := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    y : R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization.Away y S
    H : IsJacobsonRing R
    ⊢ IsJacobsonRing S
  -/
  rw [isJacobsonRing_iff_prime_eq]
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    y : R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization.Away y S
    H : IsJacobsonRing R
    ⊢ ∀ (P : Ideal S), P.IsPrime → Eq P.jacobson P
  -/
  refine fun P' hP' => le_antisymm ?_ le_jacobson
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    y : R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization.Away y S
    H : IsJacobsonRing R
    P' : Ideal S
    hP' : P'.IsPrime
    ⊢ LE.le P'.jacobson P'
  -/
  obtain ⟨hP', hPM⟩ := (IsLocalization.isPrime_iff_isPrime_disjoint (powers y) S P').mp hP'
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    y : R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization.Away y S
    H : IsJacobsonRing R
    P' : Ideal S
    hP'✝ : P'.IsPrime
    hP' : (Ideal.comap (algebraMap R S) P').IsPrime
    hPM : Disjoint ↑(Submonoid.powers y) ↑(Ideal.comap (algebraMap R S) P')
    ⊢ LE.le P'.jacobson P'
  -/
  have hP := H.out hP'.isRadical
  refine (IsLocalization.map_comap (powers y) S P'.jacobson).ge.trans
    ((map_mono ?_).trans (IsLocalization.map_comap (powers y) S P').le)
  have : sInf { I : Ideal R | comap (algebraMap R S) P' ≤ I ∧ I.IsMaximal ∧ y ∉ I } ≤
      comap (algebraMap R S) P' := by
    intro x hx
    have hxy : x * y ∈ (comap (algebraMap R S) P').jacobson := by
      rw [Ideal.jacobson, Ideal.mem_sInf]
      intro J hJ
      by_cases h : y ∈ J
      · exact J.mul_mem_left x h
      · exact J.mul_mem_right y ((mem_sInf.1 hx) ⟨hJ.left, ⟨hJ.right, h⟩⟩)
    rw [hP] at hxy
    cases' hP'.mem_or_mem hxy with hxy hxy
    · exact hxy
    · exact (hPM.le_bot ⟨Submonoid.mem_powers _, hxy⟩).elim
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    y : R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization.Away y S
    H : IsJacobsonRing R
    P' : Ideal S
    hP'✝ : P'.IsPrime
    hP' : (Ideal.comap (algebraMap R S) P').IsPrime
    hPM : Disjoint ↑(Submonoid.powers y) ↑(Ideal.comap (algebraMap R S) P')
    hP : Eq (Ideal.comap (algebraMap R S) P').jacobson (Ideal.comap (algebraMap R  …
    this : LE.le (InfSet.sInf (setOf fun I => And (LE.le (Ideal.comap (algebraMap  …
    ⊢ LE.le (Ideal.comap (algebraMap R S) P'.jacobson) (Ideal.comap (algebraMap R  …
  -/
  refine le_trans ?_ this
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    y : R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization.Away y S
    H : IsJacobsonRing R
    P' : Ideal S
    hP'✝ : P'.IsPrime
    hP' : (Ideal.comap (algebraMap R S) P').IsPrime
    hPM : Disjoint ↑(Submonoid.powers y) ↑(Ideal.comap (algebraMap R S) P')
    hP : Eq (Ideal.comap (algebraMap R S) P').jacobson (Ideal.comap (algebraMap R  …
    this : LE.le (InfSet.sInf (setOf fun I => And (LE.le (Ideal.comap (algebraMap  …
    ⊢ LE.le (Ideal.comap (algebraMap R S) P'.jacobson) (InfSet.sInf (setOf fun I = …
  -/
  rw [Ideal.jacobson, comap_sInf', sInf_eq_iInf]
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    y : R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization.Away y S
    H : IsJacobsonRing R
    P' : Ideal S
    hP'✝ : P'.IsPrime
    hP' : (Ideal.comap (algebraMap R S) P').IsPrime
    hPM : Disjoint ↑(Submonoid.powers y) ↑(Ideal.comap (algebraMap R S) P')
    hP : Eq (Ideal.comap (algebraMap R S) P').jacobson (Ideal.comap (algebraMap R  …
    this : LE.le (InfSet.sInf (setOf fun I => And (LE.le (Ideal.comap (algebraMap  …
    ⊢ LE.le (iInf fun I => iInf fun h => I) (iInf fun a => iInf fun h => a)
  -/
  refine iInf_le_iInf_of_subset fun I hI => ⟨map (algebraMap R S) I, ⟨?_, ?_⟩⟩
  · exact ⟨le_trans (le_of_eq (IsLocalization.map_comap (powers y) S P').symm) (map_mono hI.1),
      isMaximal_of_isMaximal_disjoint y _ hI.2.1 hI.2.2⟩
  · exact IsLocalization.comap_map_of_isPrime_disjoint _ S I (IsMaximal.isPrime hI.2.1)
      ((disjoint_powers_iff_not_mem y hI.2.1.isPrime.isRadical).2 hI.2.2)


lemma Subring.mem_closure_image_of {S T : Type*} [CommRing S] [CommRing T] (g : S →+* T)
    (u : Set S) (x : S) (hx : x ∈ Subring.closure u) : g x ∈ Subring.closure (g '' u) := by
  /-
    S : Type u_1
    T : Type u_2
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    g : RingHom S T
    u : Set S
    x : S
    hx : Membership.mem (Subring.closure u) x
    ⊢ Membership.mem (Subring.closure (Set.image (⇑g) u)) (g x)
  -/
  rw [Subring.mem_closure] at hx ⊢
  /-
    S : Type u_1
    T : Type u_2
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    g : RingHom S T
    u : Set S
    x : S
    hx : ∀ (S_1 : Subring S), HasSubset.Subset u ↑S_1 → Membership.mem S_1 x
    ⊢ ∀ (S_1 : Subring T), HasSubset.Subset (Set.image (⇑g) u) ↑S_1 → Membership.m …
  -/
  intro T₁ h₁
  /-
    S : Type u_1
    T : Type u_2
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    g : RingHom S T
    u : Set S
    x : S
    hx : ∀ (S_1 : Subring S), HasSubset.Subset u ↑S_1 → Membership.mem S_1 x
    T₁ : Subring T
    h₁ : HasSubset.Subset (Set.image (⇑g) u) ↑T₁
    ⊢ Membership.mem T₁ (g x)
  -/
  rw [← Subring.mem_comap]
  /-
    S : Type u_1
    T : Type u_2
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    g : RingHom S T
    u : Set S
    x : S
    hx : ∀ (S_1 : Subring S), HasSubset.Subset u ↑S_1 → Membership.mem S_1 x
    T₁ : Subring T
    h₁ : HasSubset.Subset (Set.image (⇑g) u) ↑T₁
    ⊢ Membership.mem (Subring.comap g T₁) x
  -/
  apply hx
  /-
    case a
    S : Type u_1
    T : Type u_2
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    g : RingHom S T
    u : Set S
    x : S
    hx : ∀ (S_1 : Subring S), HasSubset.Subset u ↑S_1 → Membership.mem S_1 x
    T₁ : Subring T
    h₁ : HasSubset.Subset (Set.image (⇑g) u) ↑T₁
    ⊢ HasSubset.Subset u ↑(Subring.comap g T₁)
  -/
  simp only [Subring.coe_comap, ← Set.image_subset_iff, SetLike.mem_coe]
  /-
    case a
    S : Type u_1
    T : Type u_2
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    g : RingHom S T
    u : Set S
    x : S
    hx : ∀ (S_1 : Subring S), HasSubset.Subset u ↑S_1 → Membership.mem S_1 x
    T₁ : Subring T
    h₁ : HasSubset.Subset (Set.image (⇑g) u) ↑T₁
    ⊢ HasSubset.Subset (Set.image (⇑g) u) ↑T₁
  -/
  exact h₁
  /-
    🎉 no goals
  -/

-- Porting note: move to better place

lemma mem_closure_X_union_C {R : Type*} [Ring R] (p : R[X]) :
    p ∈ Subring.closure (insert X {f | f.degree ≤ 0} : Set R[X]) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f =>  …
  -/
  refine Polynomial.induction_on p ?_ ?_ ?_
    /-
      case refine_1
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      ⊢ ∀ (a : R), Membership.mem (Subring.closure (Insert.insert Polynomial.X (setO …
    -/
  · intro r
    /-
      case refine_1
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      r : R
      ⊢ Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f =>  …
    -/
    apply Subring.subset_closure
    /-
      case refine_1.a
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      r : R
      ⊢ Membership.mem (Insert.insert Polynomial.X (setOf fun f => LE.le f.degree 0) …
    -/
    apply Set.mem_insert_of_mem
    /-
      case refine_1.a.a
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      r : R
      ⊢ Membership.mem (setOf fun f => LE.le f.degree 0) (Polynomial.C r)
    -/
    exact degree_C_le
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      ⊢ ∀ (p q : Polynomial R), Membership.mem (Subring.closure (Insert.insert Polyn …
    -/
  · intros p1 p2 h1 h2
    /-
      case refine_2
      R : Type u_1
      inst✝ : Ring R
      p p1 p2 : Polynomial R
      h1 : Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f  …
      h2 : Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f  …
      ⊢ Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f =>  …
    -/
    exact Subring.add_mem _ h1 h2
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      ⊢ ∀ (n : Nat) (a : R), Membership.mem (Subring.closure (Insert.insert Polynomi …
    -/
  · intros n r hr
    /-
      case refine_3
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      n : Nat
      r : R
      hr : Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f  …
      ⊢ Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f =>  …
    -/
    rw [pow_succ, ← mul_assoc]
    /-
      case refine_3
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      n : Nat
      r : R
      hr : Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f  …
      ⊢ Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f =>  …
    -/
    apply Subring.mul_mem _ hr
    /-
      case refine_3
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      n : Nat
      r : R
      hr : Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f  …
      ⊢ Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f =>  …
    -/
    apply Subring.subset_closure
    /-
      case refine_3.a
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      n : Nat
      r : R
      hr : Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun f  …
      ⊢ Membership.mem (Insert.insert Polynomial.X (setOf fun f => LE.le f.degree 0) …
    -/
    apply Set.mem_insert
    /-
      🎉 no goals
    -/


/-- If `I` is a prime ideal of `R[X]` and `pX ∈ I` is a non-constant polynomial,
  then the map `R →+* R[x]/I` descends to an integral map when localizing at `pX.leadingCoeff`.
  In particular `X` is integral because it satisfies `pX`, and constants are trivially integral,
  so integrality of the entire extension follows by closure under addition and multiplication. -/
theorem isIntegral_isLocalization_polynomial_quotient
    (P : Ideal R[X]) (pX : R[X]) (hpX : pX ∈ P) [Algebra (R ⧸ P.comap (C : R →+* R[X])) Rₘ]
    [IsLocalization.Away (pX.map (Ideal.Quotient.mk (P.comap (C : R →+* R[X])))).leadingCoeff Rₘ]
    [Algebra (R[X] ⧸ P) Sₘ] [IsLocalization ((Submonoid.powers (pX.map (Ideal.Quotient.mk (P.comap
      (C : R →+* R[X])))).leadingCoeff).map (quotientMap P C le_rfl) : Submonoid (R[X] ⧸ P)) Sₘ] :
    (IsLocalization.map Sₘ (quotientMap P C le_rfl) (Submonoid.powers (pX.map (Ideal.Quotient.mk
      (P.comap (C : R →+* R[X])))).leadingCoeff).le_comap_map : Rₘ →+* Sₘ).IsIntegral := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    P : Ideal (Polynomial R)
    pX : Polynomial R
    hpX : Membership.mem P pX
    inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
    inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
    inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
    inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
    ⊢ (IsLocalization.map Sₘ (Ideal.quotientMap P Polynomial.C ⋯) ⋯).IsIntegral
  -/
  let P' : Ideal R := P.comap C
  let M : Submonoid (R ⧸ P') :=
    Submonoid.powers (pX.map (Ideal.Quotient.mk (P.comap (C : R →+* R[X])))).leadingCoeff
  let M' : Submonoid (R[X] ⧸ P) :=
    (Submonoid.powers (pX.map (Ideal.Quotient.mk (P.comap (C : R →+* R[X])))).leadingCoeff).map
      (quotientMap P C le_rfl)
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    P : Ideal (Polynomial R)
    pX : Polynomial R
    hpX : Membership.mem P pX
    inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
    inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
    inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
    inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
    P' : Ideal R := Ideal.comap Polynomial.C P
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
    M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
    ⊢ (IsLocalization.map Sₘ (Ideal.quotientMap P Polynomial.C ⋯) ⋯).IsIntegral
  -/
  let φ : R ⧸ P' →+* R[X] ⧸ P := quotientMap P C le_rfl
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    P : Ideal (Polynomial R)
    pX : Polynomial R
    hpX : Membership.mem P pX
    inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
    inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
    inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
    inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
    P' : Ideal R := Ideal.comap Polynomial.C P
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
    M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    ⊢ (IsLocalization.map Sₘ (Ideal.quotientMap P Polynomial.C ⋯) ⋯).IsIntegral
  -/
  let φ' : Rₘ →+* Sₘ := IsLocalization.map Sₘ φ M.le_comap_map
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    P : Ideal (Polynomial R)
    pX : Polynomial R
    hpX : Membership.mem P pX
    inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
    inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
    inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
    inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
    P' : Ideal R := Ideal.comap Polynomial.C P
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
    M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    ⊢ (IsLocalization.map Sₘ (Ideal.quotientMap P Polynomial.C ⋯) ⋯).IsIntegral
  -/
  have hφ' : φ.comp (Ideal.Quotient.mk P') = (Ideal.Quotient.mk P).comp C := rfl
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    P : Ideal (Polynomial R)
    pX : Polynomial R
    hpX : Membership.mem P pX
    inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
    inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
    inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
    inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
    P' : Ideal R := Ideal.comap Polynomial.C P
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
    M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
    ⊢ (IsLocalization.map Sₘ (Ideal.quotientMap P Polynomial.C ⋯) ⋯).IsIntegral
  -/
  intro p
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    P : Ideal (Polynomial R)
    pX : Polynomial R
    hpX : Membership.mem P pX
    inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
    inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
    inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
    inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
    P' : Ideal R := Ideal.comap Polynomial.C P
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
    M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
    p : Sₘ
    ⊢ (IsLocalization.map Sₘ (Ideal.quotientMap P Polynomial.C ⋯) ⋯).IsIntegralEle …
  -/
  obtain ⟨⟨p', ⟨q, hq⟩⟩, hp⟩ := IsLocalization.surj M' p
  suffices φ'.IsIntegralElem (algebraMap (R[X] ⧸ P) Sₘ p') by
    obtain ⟨q', hq', rfl⟩ := hq
    obtain ⟨q'', hq''⟩ := isUnit_iff_exists_inv'.1 (IsLocalization.map_units Rₘ (⟨q', hq'⟩ : M))
    refine (hp.symm ▸ this).of_mul_unit φ' p (algebraMap (R[X] ⧸ P) Sₘ (φ q')) q'' ?_
    rw [← φ'.map_one, ← congr_arg φ' hq'', φ'.map_mul, ← φ'.comp_apply]
    simp only [φ', IsLocalization.map_comp _]
    rw [RingHom.comp_apply]
  /-
    case intro.mk.mk
    R : Type u_1
    inst✝⁶ : CommRing R
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    P : Ideal (Polynomial R)
    pX : Polynomial R
    hpX : Membership.mem P pX
    inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
    inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
    inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
    inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
    P' : Ideal R := Ideal.comap Polynomial.C P
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
    M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
    p : Sₘ
    p' q : HasQuotient.Quotient (Polynomial R) P
    hq : Membership.mem M' q
    hp : Eq (HMul.hMul p ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ)  …
    ⊢ φ'.IsIntegralElem ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) p')
  -/
  dsimp at hp
  refine @IsIntegral.of_mem_closure'' Rₘ _ Sₘ _ φ'
    ((algebraMap (R[X] ⧸ P) Sₘ).comp (Ideal.Quotient.mk P) '' insert X { p | p.degree ≤ 0 }) ?_
    ((algebraMap (R[X] ⧸ P) Sₘ) p') ?_
    /-
      case intro.mk.mk.refine_1
      R : Type u_1
      inst✝⁶ : CommRing R
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      P : Ideal (Polynomial R)
      pX : Polynomial R
      hpX : Membership.mem P pX
      inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
      inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
      inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
      inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
      P' : Ideal R := Ideal.comap Polynomial.C P
      M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
      M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
      hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
      p : Sₘ
      p' q : HasQuotient.Quotient (Polynomial R) P
      hq : Membership.mem M' q
      hp : Eq (HMul.hMul p ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ)  …
      ⊢ ∀ (x : Sₘ), Membership.mem (Set.image (⇑((algebraMap (HasQuotient.Quotient ( …
    -/
  · rintro x ⟨p, hp, rfl⟩
    /-
      case intro.mk.mk.refine_1.intro.intro
      R : Type u_1
      inst✝⁶ : CommRing R
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      P : Ideal (Polynomial R)
      pX : Polynomial R
      hpX : Membership.mem P pX
      inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
      inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
      inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
      inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
      P' : Ideal R := Ideal.comap Polynomial.C P
      M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
      M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
      hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
      p✝ : Sₘ
      p' q : HasQuotient.Quotient (Polynomial R) P
      hq : Membership.mem M' q
      hp✝ : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ …
      p : Polynomial R
      hp : Membership.mem (Insert.insert Polynomial.X (setOf fun p => LE.le p.degree …
      ⊢ φ'.IsIntegralElem (((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ). …
    -/
    simp only [Set.mem_insert_iff] at hp
    /-
      case intro.mk.mk.refine_1.intro.intro
      R : Type u_1
      inst✝⁶ : CommRing R
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      P : Ideal (Polynomial R)
      pX : Polynomial R
      hpX : Membership.mem P pX
      inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
      inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
      inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
      inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
      P' : Ideal R := Ideal.comap Polynomial.C P
      M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
      M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
      hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
      p✝ : Sₘ
      p' q : HasQuotient.Quotient (Polynomial R) P
      hq : Membership.mem M' q
      hp✝ : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ …
      p : Polynomial R
      hp : Or (Eq p Polynomial.X) (Membership.mem (setOf fun p => LE.le p.degree 0) p)
      ⊢ φ'.IsIntegralElem (((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ). …
    -/
    cases' hp with hy hy
      /-
        case intro.mk.mk.refine_1.intro.intro.inl
        R : Type u_1
        inst✝⁶ : CommRing R
        Rₘ : Type u_3
        Sₘ : Type u_4
        inst✝⁵ : CommRing Rₘ
        inst✝⁴ : CommRing Sₘ
        P : Ideal (Polynomial R)
        pX : Polynomial R
        hpX : Membership.mem P pX
        inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
        inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
        inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
        inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
        P' : Ideal R := Ideal.comap Polynomial.C P
        M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
        M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
        φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
        φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
        hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
        p✝ : Sₘ
        p' q : HasQuotient.Quotient (Polynomial R) P
        hq : Membership.mem M' q
        hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
        p : Polynomial R
        hy : Eq p Polynomial.X
        ⊢ φ'.IsIntegralElem (((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ). …
      -/
    · rw [hy]
      refine φ.isIntegralElem_localization_at_leadingCoeff ((Ideal.Quotient.mk P) X)
        (pX.map (Ideal.Quotient.mk P')) ?_ M ?_
        /-
          case intro.mk.mk.refine_1.intro.intro.inl.refine_1
          R : Type u_1
          inst✝⁶ : CommRing R
          Rₘ : Type u_3
          Sₘ : Type u_4
          inst✝⁵ : CommRing Rₘ
          inst✝⁴ : CommRing Sₘ
          P : Ideal (Polynomial R)
          pX : Polynomial R
          hpX : Membership.mem P pX
          inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
          inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
          inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
          inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
          P' : Ideal R := Ideal.comap Polynomial.C P
          M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
          M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
          φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
          φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
          hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
          p✝ : Sₘ
          p' q : HasQuotient.Quotient (Polynomial R) P
          hq : Membership.mem M' q
          hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
          p : Polynomial R
          hy : Eq p Polynomial.X
          ⊢ Eq (Polynomial.eval₂ φ ((Ideal.Quotient.mk P) Polynomial.X) (Polynomial.map  …
        -/
      · rwa [eval₂_map, hφ', ← hom_eval₂, Quotient.eq_zero_iff_mem, eval₂_C_X]
        /-
          🎉 no goals
        -/
        /-
          case intro.mk.mk.refine_1.intro.intro.inl.refine_2
          R : Type u_1
          inst✝⁶ : CommRing R
          Rₘ : Type u_3
          Sₘ : Type u_4
          inst✝⁵ : CommRing Rₘ
          inst✝⁴ : CommRing Sₘ
          P : Ideal (Polynomial R)
          pX : Polynomial R
          hpX : Membership.mem P pX
          inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
          inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
          inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
          inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
          P' : Ideal R := Ideal.comap Polynomial.C P
          M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
          M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
          φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
          φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
          hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
          p✝ : Sₘ
          p' q : HasQuotient.Quotient (Polynomial R) P
          hq : Membership.mem M' q
          hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
          p : Polynomial R
          hy : Eq p Polynomial.X
          ⊢ Membership.mem M (Polynomial.map (Ideal.Quotient.mk P') pX).leadingCoeff
        -/
      · use 1
        /-
          case h
          R : Type u_1
          inst✝⁶ : CommRing R
          Rₘ : Type u_3
          Sₘ : Type u_4
          inst✝⁵ : CommRing Rₘ
          inst✝⁴ : CommRing Sₘ
          P : Ideal (Polynomial R)
          pX : Polynomial R
          hpX : Membership.mem P pX
          inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
          inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
          inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
          inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
          P' : Ideal R := Ideal.comap Polynomial.C P
          M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
          M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
          φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
          φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
          hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
          p✝ : Sₘ
          p' q : HasQuotient.Quotient (Polynomial R) P
          hq : Membership.mem M' q
          hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
          p : Polynomial R
          hy : Eq p Polynomial.X
          ⊢ Eq ((fun x => HPow.hPow (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Poly …
        -/
        simp only [P', pow_one]
        /-
          🎉 no goals
        -/
      /-
        case intro.mk.mk.refine_1.intro.intro.inr
        R : Type u_1
        inst✝⁶ : CommRing R
        Rₘ : Type u_3
        Sₘ : Type u_4
        inst✝⁵ : CommRing Rₘ
        inst✝⁴ : CommRing Sₘ
        P : Ideal (Polynomial R)
        pX : Polynomial R
        hpX : Membership.mem P pX
        inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
        inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
        inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
        inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
        P' : Ideal R := Ideal.comap Polynomial.C P
        M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
        M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
        φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
        φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
        hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
        p✝ : Sₘ
        p' q : HasQuotient.Quotient (Polynomial R) P
        hq : Membership.mem M' q
        hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
        p : Polynomial R
        hy : Membership.mem (setOf fun p => LE.le p.degree 0) p
        ⊢ φ'.IsIntegralElem (((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ). …
      -/
    · rw [Set.mem_setOf_eq, degree_le_zero_iff] at hy
      -- Porting note: was `refine' hy.symm ▸`
      -- `⟨X - C (algebraMap _ _ ((Quotient.mk P') (p.coeff 0))), monic_X_sub_C _, _⟩`
      /-
        case intro.mk.mk.refine_1.intro.intro.inr
        R : Type u_1
        inst✝⁶ : CommRing R
        Rₘ : Type u_3
        Sₘ : Type u_4
        inst✝⁵ : CommRing Rₘ
        inst✝⁴ : CommRing Sₘ
        P : Ideal (Polynomial R)
        pX : Polynomial R
        hpX : Membership.mem P pX
        inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
        inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
        inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
        inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
        P' : Ideal R := Ideal.comap Polynomial.C P
        M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
        M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
        φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
        φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
        hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
        p✝ : Sₘ
        p' q : HasQuotient.Quotient (Polynomial R) P
        hq : Membership.mem M' q
        hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
        p : Polynomial R
        hy : Eq p (Polynomial.C (p.coeff 0))
        ⊢ φ'.IsIntegralElem (((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ). …
      -/
      rw [hy]
      /-
        case intro.mk.mk.refine_1.intro.intro.inr
        R : Type u_1
        inst✝⁶ : CommRing R
        Rₘ : Type u_3
        Sₘ : Type u_4
        inst✝⁵ : CommRing Rₘ
        inst✝⁴ : CommRing Sₘ
        P : Ideal (Polynomial R)
        pX : Polynomial R
        hpX : Membership.mem P pX
        inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
        inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
        inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
        inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
        P' : Ideal R := Ideal.comap Polynomial.C P
        M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
        M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
        φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
        φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
        hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
        p✝ : Sₘ
        p' q : HasQuotient.Quotient (Polynomial R) P
        hq : Membership.mem M' q
        hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
        p : Polynomial R
        hy : Eq p (Polynomial.C (p.coeff 0))
        ⊢ φ'.IsIntegralElem (((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ). …
      -/
      use X - C (algebraMap (R ⧸ P') Rₘ ((Ideal.Quotient.mk P') (p.coeff 0)))
      /-
        case h
        R : Type u_1
        inst✝⁶ : CommRing R
        Rₘ : Type u_3
        Sₘ : Type u_4
        inst✝⁵ : CommRing Rₘ
        inst✝⁴ : CommRing Sₘ
        P : Ideal (Polynomial R)
        pX : Polynomial R
        hpX : Membership.mem P pX
        inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
        inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
        inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
        inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
        P' : Ideal R := Ideal.comap Polynomial.C P
        M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
        M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
        φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
        φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
        hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
        p✝ : Sₘ
        p' q : HasQuotient.Quotient (Polynomial R) P
        hq : Membership.mem M' q
        hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
        p : Polynomial R
        hy : Eq p (Polynomial.C (p.coeff 0))
        ⊢ And (HSub.hSub Polynomial.X (Polynomial.C ((algebraMap (HasQuotient.Quotient …
      -/
      constructor
        /-
          case h.left
          R : Type u_1
          inst✝⁶ : CommRing R
          Rₘ : Type u_3
          Sₘ : Type u_4
          inst✝⁵ : CommRing Rₘ
          inst✝⁴ : CommRing Sₘ
          P : Ideal (Polynomial R)
          pX : Polynomial R
          hpX : Membership.mem P pX
          inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
          inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
          inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
          inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
          P' : Ideal R := Ideal.comap Polynomial.C P
          M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
          M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
          φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
          φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
          hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
          p✝ : Sₘ
          p' q : HasQuotient.Quotient (Polynomial R) P
          hq : Membership.mem M' q
          hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
          p : Polynomial R
          hy : Eq p (Polynomial.C (p.coeff 0))
          ⊢ (HSub.hSub Polynomial.X (Polynomial.C ((algebraMap (HasQuotient.Quotient R P …
        -/
      · apply monic_X_sub_C
        /-
          🎉 no goals
        -/
        /-
          case h.right
          R : Type u_1
          inst✝⁶ : CommRing R
          Rₘ : Type u_3
          Sₘ : Type u_4
          inst✝⁵ : CommRing Rₘ
          inst✝⁴ : CommRing Sₘ
          P : Ideal (Polynomial R)
          pX : Polynomial R
          hpX : Membership.mem P pX
          inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
          inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
          inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
          inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
          P' : Ideal R := Ideal.comap Polynomial.C P
          M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
          M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
          φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
          φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
          hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
          p✝ : Sₘ
          p' q : HasQuotient.Quotient (Polynomial R) P
          hq : Membership.mem M' q
          hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
          p : Polynomial R
          hy : Eq p (Polynomial.C (p.coeff 0))
          ⊢ Eq (Polynomial.eval₂ φ' (((algebraMap (HasQuotient.Quotient (Polynomial R) P …
        -/
      · simp only [eval₂_sub, eval₂_X, eval₂_C]
        /-
          case h.right
          R : Type u_1
          inst✝⁶ : CommRing R
          Rₘ : Type u_3
          Sₘ : Type u_4
          inst✝⁵ : CommRing Rₘ
          inst✝⁴ : CommRing Sₘ
          P : Ideal (Polynomial R)
          pX : Polynomial R
          hpX : Membership.mem P pX
          inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
          inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
          inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
          inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
          P' : Ideal R := Ideal.comap Polynomial.C P
          M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
          M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
          φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
          φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
          hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
          p✝ : Sₘ
          p' q : HasQuotient.Quotient (Polynomial R) P
          hq : Membership.mem M' q
          hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
          p : Polynomial R
          hy : Eq p (Polynomial.C (p.coeff 0))
          ⊢ Eq (HSub.hSub (((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ).comp …
        -/
        rw [sub_eq_zero, ← φ'.comp_apply]
        /-
          case h.right
          R : Type u_1
          inst✝⁶ : CommRing R
          Rₘ : Type u_3
          Sₘ : Type u_4
          inst✝⁵ : CommRing Rₘ
          inst✝⁴ : CommRing Sₘ
          P : Ideal (Polynomial R)
          pX : Polynomial R
          hpX : Membership.mem P pX
          inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
          inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
          inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
          inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
          P' : Ideal R := Ideal.comap Polynomial.C P
          M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
          M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
          φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
          φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
          hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
          p✝ : Sₘ
          p' q : HasQuotient.Quotient (Polynomial R) P
          hq : Membership.mem M' q
          hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
          p : Polynomial R
          hy : Eq p (Polynomial.C (p.coeff 0))
          ⊢ Eq (((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ).comp (Ideal.Quo …
        -/
        simp only [φ', IsLocalization.map_comp _]
        /-
          case h.right
          R : Type u_1
          inst✝⁶ : CommRing R
          Rₘ : Type u_3
          Sₘ : Type u_4
          inst✝⁵ : CommRing Rₘ
          inst✝⁴ : CommRing Sₘ
          P : Ideal (Polynomial R)
          pX : Polynomial R
          hpX : Membership.mem P pX
          inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
          inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
          inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
          inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
          P' : Ideal R := Ideal.comap Polynomial.C P
          M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
          M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
          φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
          φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
          hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
          p✝ : Sₘ
          p' q : HasQuotient.Quotient (Polynomial R) P
          hq : Membership.mem M' q
          hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
          p : Polynomial R
          hy : Eq p (Polynomial.C (p.coeff 0))
          ⊢ Eq (((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ).comp (Ideal.Quo …
        -/
        rfl
        /-
          🎉 no goals
        -/
    /-
      case intro.mk.mk.refine_2
      R : Type u_1
      inst✝⁶ : CommRing R
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      P : Ideal (Polynomial R)
      pX : Polynomial R
      hpX : Membership.mem P pX
      inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
      inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
      inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
      inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
      P' : Ideal R := Ideal.comap Polynomial.C P
      M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
      M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
      hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
      p : Sₘ
      p' q : HasQuotient.Quotient (Polynomial R) P
      hq : Membership.mem M' q
      hp : Eq (HMul.hMul p ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ)  …
      ⊢ Membership.mem (Subring.closure (Set.image (⇑((algebraMap (HasQuotient.Quoti …
    -/
  · obtain ⟨p, rfl⟩ := Ideal.Quotient.mk_surjective p'
    /-
      case intro.mk.mk.refine_2.intro
      R : Type u_1
      inst✝⁶ : CommRing R
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      P : Ideal (Polynomial R)
      pX : Polynomial R
      hpX : Membership.mem P pX
      inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
      inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
      inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
      inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
      P' : Ideal R := Ideal.comap Polynomial.C P
      M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
      M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
      hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
      p✝ : Sₘ
      q : HasQuotient.Quotient (Polynomial R) P
      hq : Membership.mem M' q
      p : Polynomial R
      hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
      ⊢ Membership.mem (Subring.closure (Set.image (⇑((algebraMap (HasQuotient.Quoti …
    -/
    rw [← RingHom.comp_apply]
    /-
      case intro.mk.mk.refine_2.intro
      R : Type u_1
      inst✝⁶ : CommRing R
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      P : Ideal (Polynomial R)
      pX : Polynomial R
      hpX : Membership.mem P pX
      inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
      inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
      inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
      inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
      P' : Ideal R := Ideal.comap Polynomial.C P
      M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
      M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
      hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
      p✝ : Sₘ
      q : HasQuotient.Quotient (Polynomial R) P
      hq : Membership.mem M' q
      p : Polynomial R
      hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
      ⊢ Membership.mem (Subring.closure (Set.image (⇑((algebraMap (HasQuotient.Quoti …
    -/
    apply Subring.mem_closure_image_of
    /-
      case intro.mk.mk.refine_2.intro.hx
      R : Type u_1
      inst✝⁶ : CommRing R
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      P : Ideal (Polynomial R)
      pX : Polynomial R
      hpX : Membership.mem P pX
      inst✝³ : Algebra (HasQuotient.Quotient R (Ideal.comap Polynomial.C P)) Rₘ
      inst✝² : IsLocalization.Away (Polynomial.map (Ideal.Quotient.mk (Ideal.comap P …
      inst✝¹ : Algebra (HasQuotient.Quotient (Polynomial R) P) Sₘ
      inst✝ : IsLocalization (Submonoid.map (Ideal.quotientMap P Polynomial.C ⋯) (Su …
      P' : Ideal R := Ideal.comap Polynomial.C P
      M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers (Polynomial.map  …
      M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map (Ideal …
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
      hφ' : Eq (φ.comp (Ideal.Quotient.mk P')) ((Ideal.Quotient.mk P).comp Polynomia …
      p✝ : Sₘ
      q : HasQuotient.Quotient (Polynomial R) P
      hq : Membership.mem M' q
      p : Polynomial R
      hp : Eq (HMul.hMul p✝ ((algebraMap (HasQuotient.Quotient (Polynomial R) P) Sₘ) …
      ⊢ Membership.mem (Subring.closure (Insert.insert Polynomial.X (setOf fun p =>  …
    -/
    apply Polynomial.mem_closure_X_union_C
    /-
      🎉 no goals
    -/


/-- If `f : R → S` descends to an integral map in the localization at `x`,
  and `R` is a Jacobson ring, then the intersection of all maximal ideals in `S` is trivial -/
theorem jacobson_bot_of_integral_localization
    {R : Type*} [CommRing R] [IsDomain R] [IsJacobsonRing R]
    (Rₘ Sₘ : Type*) [CommRing Rₘ] [CommRing Sₘ] (φ : R →+* S) (hφ : Function.Injective ↑φ) (x : R)
    (hx : x ≠ 0) [Algebra R Rₘ] [IsLocalization.Away x Rₘ] [Algebra S Sₘ]
    [IsLocalization ((Submonoid.powers x).map φ : Submonoid S) Sₘ]
    (hφ' :
      RingHom.IsIntegral (IsLocalization.map Sₘ φ (Submonoid.powers x).le_comap_map : Rₘ →+* Sₘ)) :
    (⊥ : Ideal S).jacobson = (⊥ : Ideal S) := by
  have hM : ((Submonoid.powers x).map φ : Submonoid S) ≤ nonZeroDivisors S :=
    map_le_nonZeroDivisors_of_injective φ hφ (powers_le_nonZeroDivisors_of_noZeroDivisors hx)
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    ⊢ Eq Bot.bot.jacobson Bot.bot
  -/
  letI : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors _ hM
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    this : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors S hM
    ⊢ Eq Bot.bot.jacobson Bot.bot
  -/
  let φ' : Rₘ →+* Sₘ := IsLocalization.map _ φ (Submonoid.powers x).le_comap_map
  suffices ∀ I : Ideal Sₘ, I.IsMaximal → (I.comap (algebraMap S Sₘ)).IsMaximal by
    have hϕ' : comap (algebraMap S Sₘ) (⊥ : Ideal Sₘ) = (⊥ : Ideal S) := by
      rw [← RingHom.ker_eq_comap_bot, ← RingHom.injective_iff_ker_eq_bot]
      exact IsLocalization.injective Sₘ hM
    have hRₘ : IsJacobsonRing Rₘ := isJacobsonRing_localization x
    have hSₘ : IsJacobsonRing Sₘ := isJacobsonRing_of_isIntegral' φ' hφ'
    refine eq_bot_iff.mpr (le_trans ?_ (le_of_eq hϕ'))
    rw [← hSₘ.out isRadical_bot_of_noZeroDivisors, comap_jacobson]
    exact sInf_le_sInf fun j hj => ⟨bot_le,
      let ⟨J, hJ⟩ := hj
      hJ.2 ▸ this J hJ.1.2⟩
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    this : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors S hM
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    ⊢ ∀ (I : Ideal Sₘ), I.IsMaximal → (Ideal.comap (algebraMap S Sₘ) I).IsMaximal
  -/
  intro I hI
  -- Remainder of the proof is pulling and pushing ideals around the square and the quotient square
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    this : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors S hM
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    I : Ideal Sₘ
    hI : I.IsMaximal
    ⊢ (Ideal.comap (algebraMap S Sₘ) I).IsMaximal
  -/
  haveI : (I.comap (algebraMap S Sₘ)).IsPrime := comap_isPrime _ I
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    this✝ : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors S hM
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    I : Ideal Sₘ
    hI : I.IsMaximal
    this : (Ideal.comap (algebraMap S Sₘ) I).IsPrime
    ⊢ (Ideal.comap (algebraMap S Sₘ) I).IsMaximal
  -/
  haveI : (I.comap φ').IsPrime := comap_isPrime φ' I
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    this✝¹ : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors S hM
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    I : Ideal Sₘ
    hI : I.IsMaximal
    this✝ : (Ideal.comap (algebraMap S Sₘ) I).IsPrime
    this : (Ideal.comap φ' I).IsPrime
    ⊢ (Ideal.comap (algebraMap S Sₘ) I).IsMaximal
  -/
  haveI : (⊥ : Ideal (S ⧸ I.comap (algebraMap S Sₘ))).IsPrime := bot_prime
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    this✝² : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors S hM
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    I : Ideal Sₘ
    hI : I.IsMaximal
    this✝¹ : (Ideal.comap (algebraMap S Sₘ) I).IsPrime
    this✝ : (Ideal.comap φ' I).IsPrime
    this : Bot.bot.IsPrime
    ⊢ (Ideal.comap (algebraMap S Sₘ) I).IsMaximal
  -/
  have hcomm : φ'.comp (algebraMap R Rₘ) = (algebraMap S Sₘ).comp φ := IsLocalization.map_comp _
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    this✝² : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors S hM
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    I : Ideal Sₘ
    hI : I.IsMaximal
    this✝¹ : (Ideal.comap (algebraMap S Sₘ) I).IsPrime
    this✝ : (Ideal.comap φ' I).IsPrime
    this : Bot.bot.IsPrime
    hcomm : Eq (φ'.comp (algebraMap R Rₘ)) ((algebraMap S Sₘ).comp φ)
    ⊢ (Ideal.comap (algebraMap S Sₘ) I).IsMaximal
  -/
  let f := quotientMap (I.comap (algebraMap S Sₘ)) φ le_rfl
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    this✝² : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors S hM
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    I : Ideal Sₘ
    hI : I.IsMaximal
    this✝¹ : (Ideal.comap (algebraMap S Sₘ) I).IsPrime
    this✝ : (Ideal.comap φ' I).IsPrime
    this : Bot.bot.IsPrime
    hcomm : Eq (φ'.comp (algebraMap R Rₘ)) ((algebraMap S Sₘ).comp φ)
    f : RingHom (HasQuotient.Quotient R (Ideal.comap φ (Ideal.comap (algebraMap S  …
    ⊢ (Ideal.comap (algebraMap S Sₘ) I).IsMaximal
  -/
  let g := quotientMap I (algebraMap S Sₘ) le_rfl
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    this✝² : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors S hM
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    I : Ideal Sₘ
    hI : I.IsMaximal
    this✝¹ : (Ideal.comap (algebraMap S Sₘ) I).IsPrime
    this✝ : (Ideal.comap φ' I).IsPrime
    this : Bot.bot.IsPrime
    hcomm : Eq (φ'.comp (algebraMap R Rₘ)) ((algebraMap S Sₘ).comp φ)
    f : RingHom (HasQuotient.Quotient R (Ideal.comap φ (Ideal.comap (algebraMap S  …
    g : RingHom (HasQuotient.Quotient S (Ideal.comap (algebraMap S Sₘ) I)) (HasQuo …
    ⊢ (Ideal.comap (algebraMap S Sₘ) I).IsMaximal
  -/
  have := isMaximal_comap_of_isIntegral_of_isMaximal' φ' hφ' I
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    this✝³ : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors S hM
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    I : Ideal Sₘ
    hI : I.IsMaximal
    this✝² : (Ideal.comap (algebraMap S Sₘ) I).IsPrime
    this✝¹ : (Ideal.comap φ' I).IsPrime
    this✝ : Bot.bot.IsPrime
    hcomm : Eq (φ'.comp (algebraMap R Rₘ)) ((algebraMap S Sₘ).comp φ)
    f : RingHom (HasQuotient.Quotient R (Ideal.comap φ (Ideal.comap (algebraMap S  …
    g : RingHom (HasQuotient.Quotient S (Ideal.comap (algebraMap S Sₘ) I)) (HasQuo …
    this : (Ideal.comap φ' I).IsMaximal
    ⊢ (Ideal.comap (algebraMap S Sₘ) I).IsMaximal
  -/
  have := ((IsLocalization.isMaximal_iff_isMaximal_disjoint Rₘ x _).1 this).left
  have : ((I.comap (algebraMap S Sₘ)).comap φ).IsMaximal := by
    rwa [comap_comap, hcomm, ← comap_comap] at this
  /-
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : IsDomain S
    R : Type u_5
    inst✝⁸ : CommRing R
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsJacobsonRing R
    Rₘ : Type u_6
    Sₘ : Type u_7
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    φ : RingHom R S
    hφ : Function.Injective ⇑φ
    x : R
    hx : Ne x 0
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization.Away x Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
    hφ' : (IsLocalization.map Sₘ φ ⋯).IsIntegral
    hM : LE.le (Submonoid.map φ (Submonoid.powers x)) (nonZeroDivisors S)
    this✝⁵ : IsDomain Sₘ := IsLocalization.isDomain_of_le_nonZeroDivisors S hM
    φ' : RingHom Rₘ Sₘ := IsLocalization.map Sₘ φ ⋯
    I : Ideal Sₘ
    hI : I.IsMaximal
    this✝⁴ : (Ideal.comap (algebraMap S Sₘ) I).IsPrime
    this✝³ : (Ideal.comap φ' I).IsPrime
    this✝² : Bot.bot.IsPrime
    hcomm : Eq (φ'.comp (algebraMap R Rₘ)) ((algebraMap S Sₘ).comp φ)
    f : RingHom (HasQuotient.Quotient R (Ideal.comap φ (Ideal.comap (algebraMap S  …
    g : RingHom (HasQuotient.Quotient S (Ideal.comap (algebraMap S Sₘ) I)) (HasQuo …
    this✝¹ : (Ideal.comap φ' I).IsMaximal
    this✝ : (Ideal.comap (algebraMap R Rₘ) (Ideal.comap φ' I)).IsMaximal
    this : (Ideal.comap φ (Ideal.comap (algebraMap S Sₘ) I)).IsMaximal
    ⊢ (Ideal.comap (algebraMap S Sₘ) I).IsMaximal
  -/
  rw [← bot_quotient_isMaximal_iff] at this ⊢
  refine isMaximal_of_isIntegral_of_isMaximal_comap' f ?_ ⊥
    ((eq_bot_iff.2 (comap_bot_le_of_injective f quotientMap_injective)).symm ▸ this)
  exact RingHom.IsIntegral.tower_bot f g quotientMap_injective
    ((comp_quotientMap_eq_of_comp_eq hcomm I).symm ▸
      (RingHom.isIntegral_of_surjective _
        (IsLocalization.surjective_quotientMap_of_maximal_of_localization (Submonoid.powers x) Rₘ
          (by rwa [comap_comap, hcomm, ← bot_quotient_isMaximal_iff]))).trans _ _ (hφ'.quotient _))


/-- Used to bootstrap the proof of `isJacobsonRing_polynomial_iff_isJacobsonRing`.
  That theorem is more general and should be used instead of this one. -/
private theorem isJacobsonRing_polynomial_of_domain (R : Type*) [CommRing R] [IsDomain R]
    [hR : IsJacobsonRing R] (P : Ideal R[X]) [IsPrime P] (hP : ∀ x : R, C x ∈ P → x = 0) :
    P.jacobson = P := by
  /-
    R : Type u_5
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    hR : IsJacobsonRing R
    P : Ideal (Polynomial R)
    inst✝ : P.IsPrime
    hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    ⊢ Eq P.jacobson P
  -/
  by_cases Pb : P = ⊥
  · exact Pb.symm ▸
      jacobson_bot_polynomial_of_jacobson_bot (hR.out isRadical_bot_of_noZeroDivisors)
    /-
      case neg
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      ⊢ Eq P.jacobson P
    -/
  · rw [jacobson_eq_iff_jacobson_quotient_eq_bot]
    /-
      case neg
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    let P' := P.comap (C : R →+* R[X])
    /-
      case neg
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    haveI : P'.IsPrime := comap_isPrime C P
    /-
      case neg
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      this : P'.IsPrime
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    haveI hR' : IsJacobsonRing (R ⧸ P') := by infer_instance
    /-
      case neg
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      this : P'.IsPrime
      hR' : IsJacobsonRing (HasQuotient.Quotient R P')
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    obtain ⟨p, pP, p0⟩ := exists_nonzero_mem_of_ne_bot Pb hP
    /-
      case neg.intro.intro
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      this : P'.IsPrime
      hR' : IsJacobsonRing (HasQuotient.Quotient R P')
      p : Polynomial R
      pP : Membership.mem P p
      p0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) p) 0
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    let x := (Polynomial.map (Ideal.Quotient.mk P') p).leadingCoeff
    /-
      case neg.intro.intro
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      this : P'.IsPrime
      hR' : IsJacobsonRing (HasQuotient.Quotient R P')
      p : Polynomial R
      pP : Membership.mem P p
      p0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) p) 0
      x : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') p).lea …
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    have hx : x ≠ 0 := by rwa [Ne, leadingCoeff_eq_zero]
    /-
      case neg.intro.intro
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      this : P'.IsPrime
      hR' : IsJacobsonRing (HasQuotient.Quotient R P')
      p : Polynomial R
      pP : Membership.mem P p
      p0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) p) 0
      x : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') p).lea …
      hx : Ne x 0
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    let φ : R ⧸ P' →+* R[X] ⧸ P := Ideal.quotientMap P (C : R →+* R[X]) le_rfl
    /-
      case neg.intro.intro
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      this : P'.IsPrime
      hR' : IsJacobsonRing (HasQuotient.Quotient R P')
      p : Polynomial R
      pP : Membership.mem P p
      p0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) p) 0
      x : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') p).lea …
      hx : Ne x 0
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    let hφ : Function.Injective ↑φ := quotientMap_injective
    /-
      case neg.intro.intro
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      this : P'.IsPrime
      hR' : IsJacobsonRing (HasQuotient.Quotient R P')
      p : Polynomial R
      pP : Membership.mem P p
      p0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) p) 0
      x : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') p).lea …
      hx : Ne x 0
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      hφ : Function.Injective ⇑φ := Ideal.quotientMap_injective
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    let Rₘ := Localization.Away x
    /-
      case neg.intro.intro
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      this : P'.IsPrime
      hR' : IsJacobsonRing (HasQuotient.Quotient R P')
      p : Polynomial R
      pP : Membership.mem P p
      p0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) p) 0
      x : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') p).lea …
      hx : Ne x 0
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      hφ : Function.Injective ⇑φ := Ideal.quotientMap_injective
      Rₘ : Type u_5 := Localization.Away x
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    let Sₘ := (Localization ((Submonoid.powers x).map φ : Submonoid (R[X] ⧸ P)))
    /-
      case neg.intro.intro
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      this : P'.IsPrime
      hR' : IsJacobsonRing (HasQuotient.Quotient R P')
      p : Polynomial R
      pP : Membership.mem P p
      p0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) p) 0
      x : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') p).lea …
      hx : Ne x 0
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      hφ : Function.Injective ⇑φ := Ideal.quotientMap_injective
      Rₘ : Type u_5 := Localization.Away x
      Sₘ : Type u_5 := Localization (Submonoid.map φ (Submonoid.powers x))
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    refine jacobson_bot_of_integral_localization (S := R[X] ⧸ P) (R := R ⧸ P') Rₘ Sₘ _ hφ _ hx ?_
    /-
      case neg.intro.intro
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      this : P'.IsPrime
      hR' : IsJacobsonRing (HasQuotient.Quotient R P')
      p : Polynomial R
      pP : Membership.mem P p
      p0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) p) 0
      x : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') p).lea …
      hx : Ne x 0
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      hφ : Function.Injective ⇑φ := Ideal.quotientMap_injective
      Rₘ : Type u_5 := Localization.Away x
      Sₘ : Type u_5 := Localization (Submonoid.map φ (Submonoid.powers x))
      ⊢ (IsLocalization.map Sₘ φ ⋯).IsIntegral
    -/
    haveI islocSₘ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ := by infer_instance
    /-
      case neg.intro.intro
      R : Type u_5
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hR : IsJacobsonRing R
      P : Ideal (Polynomial R)
      inst✝ : P.IsPrime
      hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      Pb : Not (Eq P Bot.bot)
      P' : Ideal R := Ideal.comap Polynomial.C P
      this : P'.IsPrime
      hR' : IsJacobsonRing (HasQuotient.Quotient R P')
      p : Polynomial R
      pP : Membership.mem P p
      p0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) p) 0
      x : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') p).lea …
      hx : Ne x 0
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      hφ : Function.Injective ⇑φ := Ideal.quotientMap_injective
      Rₘ : Type u_5 := Localization.Away x
      Sₘ : Type u_5 := Localization (Submonoid.map φ (Submonoid.powers x))
      islocSₘ : IsLocalization (Submonoid.map φ (Submonoid.powers x)) Sₘ
      ⊢ (IsLocalization.map Sₘ φ ⋯).IsIntegral
    -/
    exact @isIntegral_isLocalization_polynomial_quotient R _ Rₘ Sₘ _ _ P p pP _ _ _ islocSₘ
    /-
      🎉 no goals
    -/


theorem isJacobsonRing_polynomial_of_isJacobsonRing (hR : IsJacobsonRing R) :
    IsJacobsonRing R[X] := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsJacobsonRing R
    ⊢ IsJacobsonRing (Polynomial R)
  -/
  rw [isJacobsonRing_iff_prime_eq]
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsJacobsonRing R
    ⊢ ∀ (P : Ideal (Polynomial R)), P.IsPrime → Eq P.jacobson P
  -/
  intro I hI
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsJacobsonRing R
    I : Ideal (Polynomial R)
    hI : I.IsPrime
    ⊢ Eq I.jacobson I
  -/
  let R' : Subring (R[X] ⧸ I) := ((Ideal.Quotient.mk I).comp C).range
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsJacobsonRing R
    I : Ideal (Polynomial R)
    hI : I.IsPrime
    R' : Subring (HasQuotient.Quotient (Polynomial R) I) := ((Ideal.Quotient.mk I) …
    ⊢ Eq I.jacobson I
  -/
  let i : R →+* R' := ((Ideal.Quotient.mk I).comp C).rangeRestrict
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsJacobsonRing R
    I : Ideal (Polynomial R)
    hI : I.IsPrime
    R' : Subring (HasQuotient.Quotient (Polynomial R) I) := ((Ideal.Quotient.mk I) …
    i : RingHom R (Subtype fun x => Membership.mem R' x) := ((Ideal.Quotient.mk I) …
    ⊢ Eq I.jacobson I
  -/
  have hi : Function.Surjective ↑i := ((Ideal.Quotient.mk I).comp C).rangeRestrict_surjective
  have hi' : RingHom.ker (mapRingHom i) ≤ I := by
    intro f hf
    apply polynomial_mem_ideal_of_coeff_mem_ideal I f
    intro n
    replace hf := congrArg (fun g : Polynomial ((Ideal.Quotient.mk I).comp C).range => g.coeff n) hf
    change (Polynomial.map ((Ideal.Quotient.mk I).comp C).rangeRestrict f).coeff n = 0 at hf
    rw [coeff_map, Subtype.ext_iff] at hf
    rwa [mem_comap, ← Quotient.eq_zero_iff_mem, ← RingHom.comp_apply]
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsJacobsonRing R
    I : Ideal (Polynomial R)
    hI : I.IsPrime
    R' : Subring (HasQuotient.Quotient (Polynomial R) I) := ((Ideal.Quotient.mk I) …
    i : RingHom R (Subtype fun x => Membership.mem R' x) := ((Ideal.Quotient.mk I) …
    hi : Function.Surjective ⇑i
    hi' : LE.le (RingHom.ker (Polynomial.mapRingHom i)) I
    ⊢ Eq I.jacobson I
  -/
  have R'_jacob : IsJacobsonRing R' := isJacobsonRing_of_surjective ⟨i, hi⟩
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsJacobsonRing R
    I : Ideal (Polynomial R)
    hI : I.IsPrime
    R' : Subring (HasQuotient.Quotient (Polynomial R) I) := ((Ideal.Quotient.mk I) …
    i : RingHom R (Subtype fun x => Membership.mem R' x) := ((Ideal.Quotient.mk I) …
    hi : Function.Surjective ⇑i
    hi' : LE.le (RingHom.ker (Polynomial.mapRingHom i)) I
    R'_jacob : IsJacobsonRing (Subtype fun x => Membership.mem R' x)
    ⊢ Eq I.jacobson I
  -/
  let J := I.map (mapRingHom i)
  -- Porting note: moved ↓ this up a few lines, so that it can be used in the `have`
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsJacobsonRing R
    I : Ideal (Polynomial R)
    hI : I.IsPrime
    R' : Subring (HasQuotient.Quotient (Polynomial R) I) := ((Ideal.Quotient.mk I) …
    i : RingHom R (Subtype fun x => Membership.mem R' x) := ((Ideal.Quotient.mk I) …
    hi : Function.Surjective ⇑i
    hi' : LE.le (RingHom.ker (Polynomial.mapRingHom i)) I
    R'_jacob : IsJacobsonRing (Subtype fun x => Membership.mem R' x)
    J : Ideal (Polynomial (Subtype fun x => Membership.mem R' x)) := Ideal.map (Po …
    ⊢ Eq I.jacobson I
  -/
  have h_surj : Function.Surjective (mapRingHom i) := Polynomial.map_surjective i hi
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsJacobsonRing R
    I : Ideal (Polynomial R)
    hI : I.IsPrime
    R' : Subring (HasQuotient.Quotient (Polynomial R) I) := ((Ideal.Quotient.mk I) …
    i : RingHom R (Subtype fun x => Membership.mem R' x) := ((Ideal.Quotient.mk I) …
    hi : Function.Surjective ⇑i
    hi' : LE.le (RingHom.ker (Polynomial.mapRingHom i)) I
    R'_jacob : IsJacobsonRing (Subtype fun x => Membership.mem R' x)
    J : Ideal (Polynomial (Subtype fun x => Membership.mem R' x)) := Ideal.map (Po …
    h_surj : Function.Surjective ⇑(Polynomial.mapRingHom i)
    ⊢ Eq I.jacobson I
  -/
  have : IsPrime J := map_isPrime_of_surjective h_surj hi'
  suffices h : J.jacobson = J by
    replace h := congrArg (comap (Polynomial.mapRingHom i)) h
    rw [← map_jacobson_of_surjective h_surj hi', comap_map_of_surjective _ h_surj,
      comap_map_of_surjective _ h_surj] at h
    refine le_antisymm ?_ le_jacobson
    exact le_trans (le_sup_of_le_left le_rfl) (le_trans (le_of_eq h) (sup_le le_rfl hi'))
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsJacobsonRing R
    I : Ideal (Polynomial R)
    hI : I.IsPrime
    R' : Subring (HasQuotient.Quotient (Polynomial R) I) := ((Ideal.Quotient.mk I) …
    i : RingHom R (Subtype fun x => Membership.mem R' x) := ((Ideal.Quotient.mk I) …
    hi : Function.Surjective ⇑i
    hi' : LE.le (RingHom.ker (Polynomial.mapRingHom i)) I
    R'_jacob : IsJacobsonRing (Subtype fun x => Membership.mem R' x)
    J : Ideal (Polynomial (Subtype fun x => Membership.mem R' x)) := Ideal.map (Po …
    h_surj : Function.Surjective ⇑(Polynomial.mapRingHom i)
    this : J.IsPrime
    ⊢ Eq J.jacobson J
  -/
  apply isJacobsonRing_polynomial_of_domain R' J
  /-
    R : Type u_1
    inst✝ : CommRing R
    hR : IsJacobsonRing R
    I : Ideal (Polynomial R)
    hI : I.IsPrime
    R' : Subring (HasQuotient.Quotient (Polynomial R) I) := ((Ideal.Quotient.mk I) …
    i : RingHom R (Subtype fun x => Membership.mem R' x) := ((Ideal.Quotient.mk I) …
    hi : Function.Surjective ⇑i
    hi' : LE.le (RingHom.ker (Polynomial.mapRingHom i)) I
    R'_jacob : IsJacobsonRing (Subtype fun x => Membership.mem R' x)
    J : Ideal (Polynomial (Subtype fun x => Membership.mem R' x)) := Ideal.map (Po …
    h_surj : Function.Surjective ⇑(Polynomial.mapRingHom i)
    this : J.IsPrime
    ⊢ ∀ (x : Subtype fun x => Membership.mem R' x), Membership.mem J (Polynomial.C …
  -/
  exact eq_zero_of_polynomial_mem_map_range I
  /-
    🎉 no goals
  -/


theorem isJacobsonRing_polynomial_iff_isJacobsonRing : IsJacobsonRing R[X] ↔ IsJacobsonRing R := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Iff (IsJacobsonRing (Polynomial R)) (IsJacobsonRing R)
  -/
  refine ⟨?_, isJacobsonRing_polynomial_of_isJacobsonRing⟩
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ IsJacobsonRing (Polynomial R) → IsJacobsonRing R
  -/
  intro H
  exact isJacobsonRing_of_surjective ⟨eval₂RingHom (RingHom.id _) 1, fun x =>
    ⟨C x, by simp only [coe_eval₂RingHom, RingHom.id_apply, eval₂_C]⟩⟩


instance [IsJacobsonRing R] : IsJacobsonRing R[X] :=
  isJacobsonRing_polynomial_iff_isJacobsonRing.mpr ‹IsJacobsonRing R›


theorem isMaximal_comap_C_of_isMaximal [IsJacobsonRing R] [Nontrivial R]
    (hP' : ∀ x : R, C x ∈ P → x = 0) :
    IsMaximal (comap (C : R →+* R[X]) P : Ideal R) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    ⊢ (Ideal.comap Polynomial.C P).IsMaximal
  -/
  let P' := comap (C : R →+* R[X]) P
  /-
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    ⊢ (Ideal.comap Polynomial.C P).IsMaximal
  -/
  haveI hP'_prime : P'.IsPrime := comap_isPrime C P
  obtain ⟨⟨m, hmem_P⟩, hm⟩ :=
    Submodule.nonzero_mem_of_bot_lt (bot_lt_of_maximal P polynomial_not_isField)
  have hm' : m ≠ 0 := by
    simpa [Submodule.coe_eq_zero] using hm
  /-
    case intro.mk
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    hP'_prime : P'.IsPrime
    m : Polynomial R
    hmem_P : Membership.mem P m
    hm : Ne ⟨m, hmem_P⟩ 0
    hm' : Ne m 0
    ⊢ (Ideal.comap Polynomial.C P).IsMaximal
  -/
  let φ : R ⧸ P' →+* R[X] ⧸ P := quotientMap P (C : R →+* R[X]) le_rfl
  /-
    case intro.mk
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    hP'_prime : P'.IsPrime
    m : Polynomial R
    hmem_P : Membership.mem P m
    hm : Ne ⟨m, hmem_P⟩ 0
    hm' : Ne m 0
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    ⊢ (Ideal.comap Polynomial.C P).IsMaximal
  -/
  let a : R ⧸ P' := (m.map (Ideal.Quotient.mk P')).leadingCoeff
  /-
    case intro.mk
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    hP'_prime : P'.IsPrime
    m : Polynomial R
    hmem_P : Membership.mem P m
    hm : Ne ⟨m, hmem_P⟩ 0
    hm' : Ne m 0
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') m).lea …
    ⊢ (Ideal.comap Polynomial.C P).IsMaximal
  -/
  let M : Submonoid (R ⧸ P') := Submonoid.powers a
  /-
    case intro.mk
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    hP'_prime : P'.IsPrime
    m : Polynomial R
    hmem_P : Membership.mem P m
    hm : Ne ⟨m, hmem_P⟩ 0
    hm' : Ne m 0
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') m).lea …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    ⊢ (Ideal.comap Polynomial.C P).IsMaximal
  -/
  rw [← bot_quotient_isMaximal_iff]
  have hp0 : a ≠ 0 := fun hp0' =>
    hm' <| map_injective (Ideal.Quotient.mk (P.comap (C : R →+* R[X]) : Ideal R))
      ((injective_iff_map_eq_zero (Ideal.Quotient.mk (P.comap (C : R →+* R[X]) : Ideal R))).2
        fun x hx => by
          rwa [Quotient.eq_zero_iff_mem, (by rwa [eq_bot_iff] : (P.comap C : Ideal R) = ⊥)] at hx)
        (by simpa only [a, leadingCoeff_eq_zero, Polynomial.map_zero] using hp0')
  /-
    case intro.mk
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    hP'_prime : P'.IsPrime
    m : Polynomial R
    hmem_P : Membership.mem P m
    hm : Ne ⟨m, hmem_P⟩ 0
    hm' : Ne m 0
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') m).lea …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    hp0 : Ne a 0
    ⊢ Bot.bot.IsMaximal
  -/
  have hM : (0 : R ⧸ P') ∉ M := fun ⟨n, hn⟩ => hp0 (pow_eq_zero hn)
  suffices (⊥ : Ideal (Localization M)).IsMaximal by
    rw [← IsLocalization.comap_map_of_isPrime_disjoint M (Localization M) ⊥ bot_prime
      (disjoint_iff_inf_le.mpr fun x hx => hM (hx.2 ▸ hx.1))]
    exact ((IsLocalization.isMaximal_iff_isMaximal_disjoint (Localization M) a _).mp
      (by rwa [Ideal.map_bot])).1
  /-
    case intro.mk
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    hP'_prime : P'.IsPrime
    m : Polynomial R
    hmem_P : Membership.mem P m
    hm : Ne ⟨m, hmem_P⟩ 0
    hm' : Ne m 0
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') m).lea …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    hp0 : Ne a 0
    hM : Not (Membership.mem M 0)
    ⊢ Bot.bot.IsMaximal
  -/
  let M' : Submonoid (R[X] ⧸ P) := M.map φ
  have hM' : (0 : R[X] ⧸ P) ∉ M' := fun ⟨z, hz⟩ =>
    hM (quotientMap_injective (_root_.trans hz.2 φ.map_zero.symm) ▸ hz.1)
  haveI : IsDomain (Localization M') :=
    IsLocalization.isDomain_localization (le_nonZeroDivisors_of_noZeroDivisors hM')
  suffices (⊥ : Ideal (Localization M')).IsMaximal by
    rw [le_antisymm bot_le (comap_bot_le_of_injective _
      (IsLocalization.map_injective_of_injective M (Localization M) (Localization M')
        quotientMap_injective))]
    refine isMaximal_comap_of_isIntegral_of_isMaximal' _ ?_ ⊥
    have isloc : IsLocalization (Submonoid.map φ M) (Localization M') := by infer_instance
    exact @isIntegral_isLocalization_polynomial_quotient R _
      (Localization M) (Localization M') _ _ P m hmem_P _ _ _ isloc
  rw [(map_bot.symm :
    (⊥ : Ideal (Localization M')) = Ideal.map (algebraMap (R[X] ⧸ P) (Localization M')) ⊥)]
  /-
    case intro.mk
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    hP'_prime : P'.IsPrime
    m : Polynomial R
    hmem_P : Membership.mem P m
    hm : Ne ⟨m, hmem_P⟩ 0
    hm' : Ne m 0
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') m).lea …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    hp0 : Ne a 0
    hM : Not (Membership.mem M 0)
    M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map φ M
    hM' : Not (Membership.mem M' 0)
    this : IsDomain (Localization M')
    ⊢ (Ideal.map (algebraMap (HasQuotient.Quotient (Polynomial R) P) (Localization …
  -/
  let bot_maximal := (bot_quotient_isMaximal_iff _).mpr hP
  /-
    case intro.mk
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    hP'_prime : P'.IsPrime
    m : Polynomial R
    hmem_P : Membership.mem P m
    hm : Ne ⟨m, hmem_P⟩ 0
    hm' : Ne m 0
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') m).lea …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    hp0 : Ne a 0
    hM : Not (Membership.mem M 0)
    M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map φ M
    hM' : Not (Membership.mem M' 0)
    this : IsDomain (Localization M')
    bot_maximal : Bot.bot.IsMaximal := (Ideal.bot_quotient_isMaximal_iff P).mpr hP
    ⊢ (Ideal.map (algebraMap (HasQuotient.Quotient (Polynomial R) P) (Localization …
  -/
  refine bot_maximal.map_bijective (algebraMap (R[X] ⧸ P) (Localization M')) ?_
  /-
    case intro.mk
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    hP'_prime : P'.IsPrime
    m : Polynomial R
    hmem_P : Membership.mem P m
    hm : Ne ⟨m, hmem_P⟩ 0
    hm' : Ne m 0
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') m).lea …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    hp0 : Ne a 0
    hM : Not (Membership.mem M 0)
    M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map φ M
    hM' : Not (Membership.mem M' 0)
    this : IsDomain (Localization M')
    bot_maximal : Bot.bot.IsMaximal := (Ideal.bot_quotient_isMaximal_iff P).mpr hP
    ⊢ Function.Bijective ⇑(algebraMap (HasQuotient.Quotient (Polynomial R) P) (Loc …
  -/
  apply IsField.localization_map_bijective hM'
  /-
    case intro.mk.hR
    R : Type u_1
    inst✝² : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝¹ : IsJacobsonRing R
    inst✝ : Nontrivial R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    hP'_prime : P'.IsPrime
    m : Polynomial R
    hmem_P : Membership.mem P m
    hm : Ne ⟨m, hmem_P⟩ 0
    hm' : Ne m 0
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') m).lea …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    hp0 : Ne a 0
    hM : Not (Membership.mem M 0)
    M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map φ M
    hM' : Not (Membership.mem M' 0)
    this : IsDomain (Localization M')
    bot_maximal : Bot.bot.IsMaximal := (Ideal.bot_quotient_isMaximal_iff P).mpr hP
    ⊢ IsField (HasQuotient.Quotient (Polynomial R) P)
  -/
  rwa [← Quotient.maximal_ideal_iff_isField_quotient, ← bot_quotient_isMaximal_iff]
  /-
    🎉 no goals
  -/


/-- Used to bootstrap the more general `quotient_mk_comp_C_isIntegral_of_jacobson` -/
private theorem quotient_mk_comp_C_isIntegral_of_jacobson' [Nontrivial R] (hR : IsJacobsonRing R)
    (hP' : ∀ x : R, C x ∈ P → x = 0) :
    ((Ideal.Quotient.mk P).comp C : R →+* R[X] ⧸ P).IsIntegral := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : Nontrivial R
    hR : IsJacobsonRing R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    ⊢ ((Ideal.Quotient.mk P).comp Polynomial.C).IsIntegral
  -/
  refine (isIntegral_quotientMap_iff _).mp ?_
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : Nontrivial R
    hR : IsJacobsonRing R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    ⊢ (Ideal.quotientMap P Polynomial.C ⋯).IsIntegral
  -/
  let P' : Ideal R := P.comap C
  obtain ⟨pX, hpX, hp0⟩ :=
    exists_nonzero_mem_of_ne_bot (ne_of_lt (bot_lt_of_maximal P polynomial_not_isField)).symm hP'
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : Nontrivial R
    hR : IsJacobsonRing R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    pX : Polynomial R
    hpX : Membership.mem P pX
    hp0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) pX) 0
    ⊢ (Ideal.quotientMap P Polynomial.C ⋯).IsIntegral
  -/
  let a : R ⧸ P' := (pX.map (Ideal.Quotient.mk P')).leadingCoeff
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : Nontrivial R
    hR : IsJacobsonRing R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    pX : Polynomial R
    hpX : Membership.mem P pX
    hp0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) pX) 0
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') pX).le …
    ⊢ (Ideal.quotientMap P Polynomial.C ⋯).IsIntegral
  -/
  let M : Submonoid (R ⧸ P') := Submonoid.powers a
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : Nontrivial R
    hR : IsJacobsonRing R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    pX : Polynomial R
    hpX : Membership.mem P pX
    hp0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) pX) 0
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') pX).le …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    ⊢ (Ideal.quotientMap P Polynomial.C ⋯).IsIntegral
  -/
  let φ : R ⧸ P' →+* R[X] ⧸ P := quotientMap P C le_rfl
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : Nontrivial R
    hR : IsJacobsonRing R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    pX : Polynomial R
    hpX : Membership.mem P pX
    hp0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) pX) 0
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') pX).le …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    ⊢ (Ideal.quotientMap P Polynomial.C ⋯).IsIntegral
  -/
  haveI hP'_prime : P'.IsPrime := comap_isPrime C P
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : Nontrivial R
    hR : IsJacobsonRing R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    pX : Polynomial R
    hpX : Membership.mem P pX
    hp0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) pX) 0
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') pX).le …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    hP'_prime : P'.IsPrime
    ⊢ (Ideal.quotientMap P Polynomial.C ⋯).IsIntegral
  -/
  have hM : (0 : R ⧸ P') ∉ M := fun ⟨n, hn⟩ => hp0 <| leadingCoeff_eq_zero.mp (pow_eq_zero hn)
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : Nontrivial R
    hR : IsJacobsonRing R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    pX : Polynomial R
    hpX : Membership.mem P pX
    hp0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) pX) 0
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') pX).le …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    hP'_prime : P'.IsPrime
    hM : Not (Membership.mem M 0)
    ⊢ (Ideal.quotientMap P Polynomial.C ⋯).IsIntegral
  -/
  let M' : Submonoid (R[X] ⧸ P) := M.map φ
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : Nontrivial R
    hR : IsJacobsonRing R
    hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    P' : Ideal R := Ideal.comap Polynomial.C P
    pX : Polynomial R
    hpX : Membership.mem P pX
    hp0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) pX) 0
    a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') pX).le …
    M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
    φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
    hP'_prime : P'.IsPrime
    hM : Not (Membership.mem M 0)
    M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map φ M
    ⊢ (Ideal.quotientMap P Polynomial.C ⋯).IsIntegral
  -/
  refine RingHom.IsIntegral.tower_bot φ (algebraMap _ (Localization M')) ?_ ?_
  · refine IsLocalization.injective (Localization M')
      (show M' ≤ _ from le_nonZeroDivisors_of_noZeroDivisors fun hM' => hM ?_)
    exact
      let ⟨z, zM, z0⟩ := hM'
      quotientMap_injective (_root_.trans z0 φ.map_zero.symm) ▸ zM
  · suffices RingHom.comp (algebraMap (R[X] ⧸ P) (Localization M')) φ =
      (IsLocalization.map (Localization M') φ M.le_comap_map).comp
        (algebraMap (R ⧸ P') (Localization M)) by
      rw [this]
      refine RingHom.IsIntegral.trans (algebraMap (R ⧸ P') (Localization M))
        (IsLocalization.map (Localization M') φ M.le_comap_map) ?_ ?_
      · exact (algebraMap (R ⧸ P') (Localization M)).isIntegral_of_surjective
          (IsField.localization_map_bijective hM ((Quotient.maximal_ideal_iff_isField_quotient _).mp
            (isMaximal_comap_C_of_isMaximal P hP'))).2
      · -- `convert` here is faster than `exact`, and this proof is near the time limit.
        -- convert isIntegral_isLocalization_polynomial_quotient P pX hpX
        have isloc : IsLocalization M' (Localization M') := by infer_instance
        exact @isIntegral_isLocalization_polynomial_quotient R _
          (Localization M) (Localization M') _ _ P pX hpX _ _ _ isloc
    /-
      case intro.intro.refine_2
      R : Type u_1
      inst✝¹ : CommRing R
      P : Ideal (Polynomial R)
      hP : P.IsMaximal
      inst✝ : Nontrivial R
      hR : IsJacobsonRing R
      hP' : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
      P' : Ideal R := Ideal.comap Polynomial.C P
      pX : Polynomial R
      hpX : Membership.mem P pX
      hp0 : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) pX) 0
      a : HasQuotient.Quotient R P' := (Polynomial.map (Ideal.Quotient.mk P') pX).le …
      M : Submonoid (HasQuotient.Quotient R P') := Submonoid.powers a
      φ : RingHom (HasQuotient.Quotient R P') (HasQuotient.Quotient (Polynomial R) P …
      hP'_prime : P'.IsPrime
      hM : Not (Membership.mem M 0)
      M' : Submonoid (HasQuotient.Quotient (Polynomial R) P) := Submonoid.map φ M
      ⊢ Eq ((algebraMap (HasQuotient.Quotient (Polynomial R) P) (Localization M')).c …
    -/
    rw [IsLocalization.map_comp M.le_comap_map]
    /-
      🎉 no goals
    -/


/-- If `R` is a Jacobson ring, and `P` is a maximal ideal of `R[X]`,
  then `R → R[X]/P` is an integral map. -/
theorem quotient_mk_comp_C_isIntegral_of_isJacobsonRing :
    ((Ideal.Quotient.mk P).comp C : R →+* R[X] ⧸ P).IsIntegral := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    ⊢ ((Ideal.Quotient.mk P).comp Polynomial.C).IsIntegral
  -/
  let P' : Ideal R := P.comap C
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    P' : Ideal R := Ideal.comap Polynomial.C P
    ⊢ ((Ideal.Quotient.mk P).comp Polynomial.C).IsIntegral
  -/
  haveI : P'.IsPrime := comap_isPrime C P
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    P' : Ideal R := Ideal.comap Polynomial.C P
    this : P'.IsPrime
    ⊢ ((Ideal.Quotient.mk P).comp Polynomial.C).IsIntegral
  -/
  let f : R[X] →+* Polynomial (R ⧸ P') := Polynomial.mapRingHom (Ideal.Quotient.mk P')
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    P' : Ideal R := Ideal.comap Polynomial.C P
    this : P'.IsPrime
    f : RingHom (Polynomial R) (Polynomial (HasQuotient.Quotient R P')) := Polynom …
    ⊢ ((Ideal.Quotient.mk P).comp Polynomial.C).IsIntegral
  -/
  have hf : Function.Surjective ↑f := map_surjective (Ideal.Quotient.mk P') Quotient.mk_surjective
  have hPJ : P = (P.map f).comap f := by
    rw [comap_map_of_surjective _ hf]
    refine le_antisymm (le_sup_of_le_left le_rfl) (sup_le le_rfl ?_)
    refine fun p hp =>
      polynomial_mem_ideal_of_coeff_mem_ideal P p fun n => Quotient.eq_zero_iff_mem.mp ?_
    simpa only [f, coeff_map, coe_mapRingHom] using (Polynomial.ext_iff.mp hp) n
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    P' : Ideal R := Ideal.comap Polynomial.C P
    this : P'.IsPrime
    f : RingHom (Polynomial R) (Polynomial (HasQuotient.Quotient R P')) := Polynom …
    hf : Function.Surjective ⇑f
    hPJ : Eq P (Ideal.comap f (Ideal.map f P))
    ⊢ ((Ideal.Quotient.mk P).comp Polynomial.C).IsIntegral
  -/
  refine RingHom.IsIntegral.tower_bot _ _ (injective_quotient_le_comap_map P) ?_
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    P' : Ideal R := Ideal.comap Polynomial.C P
    this : P'.IsPrime
    f : RingHom (Polynomial R) (Polynomial (HasQuotient.Quotient R P')) := Polynom …
    hf : Function.Surjective ⇑f
    hPJ : Eq P (Ideal.comap f (Ideal.map f P))
    ⊢ ((Ideal.quotientMap (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.mk (Id …
  -/
  rw [← quotient_mk_maps_eq]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    P' : Ideal R := Ideal.comap Polynomial.C P
    this : P'.IsPrime
    f : RingHom (Polynomial R) (Polynomial (HasQuotient.Quotient R P')) := Polynom …
    hf : Function.Surjective ⇑f
    hPJ : Eq P (Ideal.comap f (Ideal.map f P))
    ⊢ (((Ideal.Quotient.mk (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.mk (I …
  -/
  refine ((Ideal.Quotient.mk P').isIntegral_of_surjective Quotient.mk_surjective).trans _ _ ?_
  have : IsMaximal (Ideal.map (mapRingHom (Ideal.Quotient.mk (comap C P))) P) :=
    Or.recOn (map_eq_top_or_isMaximal_of_surjective f hf hP)
      (fun h => absurd (_root_.trans (h ▸ hPJ : P = comap f ⊤) comap_top : P = ⊤) hP.ne_top) id
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    P' : Ideal R := Ideal.comap Polynomial.C P
    this✝ : P'.IsPrime
    f : RingHom (Polynomial R) (Polynomial (HasQuotient.Quotient R P')) := Polynom …
    hf : Function.Surjective ⇑f
    hPJ : Eq P (Ideal.comap f (Ideal.map f P))
    this : (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.mk (Ideal.comap Polyn …
    ⊢ ((Ideal.Quotient.mk (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.mk (Id …
  -/
  apply quotient_mk_comp_C_isIntegral_of_jacobson' _ ?_ (fun x hx => ?_)
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    P' : Ideal R := Ideal.comap Polynomial.C P
    this✝ : P'.IsPrime
    f : RingHom (Polynomial R) (Polynomial (HasQuotient.Quotient R P')) := Polynom …
    hf : Function.Surjective ⇑f
    hPJ : Eq P (Ideal.comap f (Ideal.map f P))
    this : (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.mk (Ideal.comap Polyn …
    ⊢ IsJacobsonRing (HasQuotient.Quotient R P')
  -/
  any_goals exact isJacobsonRing_quotient
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    P' : Ideal R := Ideal.comap Polynomial.C P
    this✝ : P'.IsPrime
    f : RingHom (Polynomial R) (Polynomial (HasQuotient.Quotient R P')) := Polynom …
    hf : Function.Surjective ⇑f
    hPJ : Eq P (Ideal.comap f (Ideal.map f P))
    this : (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.mk (Ideal.comap Polyn …
    x : HasQuotient.Quotient R P'
    hx : Membership.mem (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.mk (Idea …
    ⊢ Eq x 0
  -/
  obtain ⟨z, rfl⟩ := Ideal.Quotient.mk_surjective x
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    P' : Ideal R := Ideal.comap Polynomial.C P
    this✝ : P'.IsPrime
    f : RingHom (Polynomial R) (Polynomial (HasQuotient.Quotient R P')) := Polynom …
    hf : Function.Surjective ⇑f
    hPJ : Eq P (Ideal.comap f (Ideal.map f P))
    this : (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.mk (Ideal.comap Polyn …
    z : R
    hx : Membership.mem (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.mk (Idea …
    ⊢ Eq ((Ideal.Quotient.mk P') z) 0
  -/
  rwa [Quotient.eq_zero_iff_mem, mem_comap, hPJ, mem_comap, coe_mapRingHom, map_C]
  /-
    🎉 no goals
  -/


theorem isMaximal_comap_C_of_isJacobsonRing : (P.comap (C : R →+* R[X])).IsMaximal := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    ⊢ (Ideal.comap Polynomial.C P).IsMaximal
  -/
  rw [← @mk_ker _ _ P, RingHom.ker_eq_comap_bot, comap_comap]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    P : Ideal (Polynomial R)
    hP : P.IsMaximal
    inst✝ : IsJacobsonRing R
    ⊢ (Ideal.comap ((Ideal.Quotient.mk P).comp Polynomial.C) Bot.bot).IsMaximal
  -/
  have := (bot_quotient_isMaximal_iff _).mpr hP
  exact isMaximal_comap_of_isIntegral_of_isMaximal' _
    (quotient_mk_comp_C_isIntegral_of_isJacobsonRing P) ⊥


theorem comp_C_integral_of_surjective_of_isJacobsonRing {S : Type*} [Field S] (f : R[X] →+* S)
    (hf : Function.Surjective ↑f) : (f.comp C).IsIntegral := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsJacobsonRing R
    S : Type u_2
    inst✝ : Field S
    f : RingHom (Polynomial R) S
    hf : Function.Surjective ⇑f
    ⊢ (f.comp Polynomial.C).IsIntegral
  -/
  haveI : f.ker.IsMaximal := RingHom.ker_isMaximal_of_surjective f hf
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsJacobsonRing R
    S : Type u_2
    inst✝ : Field S
    f : RingHom (Polynomial R) S
    hf : Function.Surjective ⇑f
    this : (RingHom.ker f).IsMaximal
    ⊢ (f.comp Polynomial.C).IsIntegral
  -/
  let g : R[X] ⧸ (RingHom.ker f) →+* S := Ideal.Quotient.lift (RingHom.ker f) f fun _ h => h
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsJacobsonRing R
    S : Type u_2
    inst✝ : Field S
    f : RingHom (Polynomial R) S
    hf : Function.Surjective ⇑f
    this : (RingHom.ker f).IsMaximal
    g : RingHom (HasQuotient.Quotient (Polynomial R) (RingHom.ker f)) S := Ideal.Q …
    ⊢ (f.comp Polynomial.C).IsIntegral
  -/
  have hfg : g.comp (Ideal.Quotient.mk (RingHom.ker f)) = f := ringHom_ext' rfl rfl
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsJacobsonRing R
    S : Type u_2
    inst✝ : Field S
    f : RingHom (Polynomial R) S
    hf : Function.Surjective ⇑f
    this : (RingHom.ker f).IsMaximal
    g : RingHom (HasQuotient.Quotient (Polynomial R) (RingHom.ker f)) S := Ideal.Q …
    hfg : Eq (g.comp (Ideal.Quotient.mk (RingHom.ker f))) f
    ⊢ (f.comp Polynomial.C).IsIntegral
  -/
  rw [← hfg, RingHom.comp_assoc]
  refine (quotient_mk_comp_C_isIntegral_of_isJacobsonRing (RingHom.ker f)).trans _ g
    (g.isIntegral_of_surjective ?_)
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsJacobsonRing R
    S : Type u_2
    inst✝ : Field S
    f : RingHom (Polynomial R) S
    hf : Function.Surjective ⇑f
    this : (RingHom.ker f).IsMaximal
    g : RingHom (HasQuotient.Quotient (Polynomial R) (RingHom.ker f)) S := Ideal.Q …
    hfg : Eq (g.comp (Ideal.Quotient.mk (RingHom.ker f))) f
    ⊢ Function.Surjective ⇑g
  -/
  rw [← hfg] at hf
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsJacobsonRing R
    S : Type u_2
    inst✝ : Field S
    f : RingHom (Polynomial R) S
    this : (RingHom.ker f).IsMaximal
    g : RingHom (HasQuotient.Quotient (Polynomial R) (RingHom.ker f)) S := Ideal.Q …
    hf : Function.Surjective ⇑(g.comp (Ideal.Quotient.mk (RingHom.ker f)))
    hfg : Eq (g.comp (Ideal.Quotient.mk (RingHom.ker f))) f
    ⊢ Function.Surjective ⇑g
  -/
  norm_num at hf
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsJacobsonRing R
    S : Type u_2
    inst✝ : Field S
    f : RingHom (Polynomial R) S
    this : (RingHom.ker f).IsMaximal
    g : RingHom (HasQuotient.Quotient (Polynomial R) (RingHom.ker f)) S := Ideal.Q …
    hf : Function.Surjective (Function.comp ⇑g ⇑(Ideal.Quotient.mk (RingHom.ker f)))
    hfg : Eq (g.comp (Ideal.Quotient.mk (RingHom.ker f))) f
    ⊢ Function.Surjective ⇑g
  -/
  exact Function.Surjective.of_comp hf
  /-
    🎉 no goals
  -/


theorem isJacobsonRing_MvPolynomial_fin {R : Type u} [CommRing R] [H : IsJacobsonRing R] :
    ∀ n : ℕ, IsJacobsonRing (MvPolynomial (Fin n) R)
  | 0 => (isJacobsonRing_iso ((renameEquiv R (Equiv.equivPEmpty (Fin 0))).toRingEquiv.trans
    (isEmptyRingEquiv R PEmpty.{u+1}))).mpr H
  | n + 1 => (isJacobsonRing_iso (finSuccEquiv R n).toRingEquiv).2
    (Polynomial.isJacobsonRing_polynomial_iff_isJacobsonRing.2 (isJacobsonRing_MvPolynomial_fin n))


/-- General form of the Nullstellensatz for Jacobson rings, since in a Jacobson ring we have
  `Inf {P maximal | P ≥ I} = Inf {P prime | P ≥ I} = I.radical`. Fields are always Jacobson,
  and in that special case this is (most of) the classical Nullstellensatz,
  since `I(V(I))` is the intersection of maximal ideals containing `I`, which is then `I.radical` -/
instance isJacobsonRing {R : Type*} [CommRing R] {ι : Type*} [Finite ι] [IsJacobsonRing R] :
    IsJacobsonRing (MvPolynomial ι R) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_2
    inst✝¹ : Finite ι
    inst✝ : IsJacobsonRing R
    ⊢ IsJacobsonRing (MvPolynomial ι R)
  -/
  cases nonempty_fintype ι
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_2
    inst✝¹ : Finite ι
    inst✝ : IsJacobsonRing R
    val✝ : Fintype ι
    ⊢ IsJacobsonRing (MvPolynomial ι R)
  -/
  haveI := Classical.decEq ι
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_2
    inst✝¹ : Finite ι
    inst✝ : IsJacobsonRing R
    val✝ : Fintype ι
    this : DecidableEq ι
    ⊢ IsJacobsonRing (MvPolynomial ι R)
  -/
  let e := Fintype.equivFin ι
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_2
    inst✝¹ : Finite ι
    inst✝ : IsJacobsonRing R
    val✝ : Fintype ι
    this : DecidableEq ι
    e : Equiv ι (Fin (Fintype.card ι)) := Fintype.equivFin ι
    ⊢ IsJacobsonRing (MvPolynomial ι R)
  -/
  rw [isJacobsonRing_iso (renameEquiv R e).toRingEquiv]
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_2
    inst✝¹ : Finite ι
    inst✝ : IsJacobsonRing R
    val✝ : Fintype ι
    this : DecidableEq ι
    e : Equiv ι (Fin (Fintype.card ι)) := Fintype.equivFin ι
    ⊢ IsJacobsonRing (MvPolynomial (Fin (Fintype.card ι)) R)
  -/
  exact isJacobsonRing_MvPolynomial_fin _
  /-
    🎉 no goals
  -/


/-- The constant coefficient as an R-linear morphism -/
private noncomputable def Cₐ (R : Type u) (S : Type v)
    [CommRing R] [CommRing S] [Algebra R S] : S →ₐ[R] S[X] :=
                                               /-
                                                 n : Nat
                                                 R : Type u
                                                 S : Type v
                                                 inst✝² : CommRing R
                                                 inst✝¹ : CommRing S
                                                 inst✝ : Algebra R S
                                                 r : R
                                                 ⊢ Eq ((↑↑__src✝).toFun ((algebraMap R S) r)) ((algebraMap R (Polynomial S)) r)
                                               -/
  { Polynomial.C with commutes' := fun r => by rfl }
                                               /-
                                                 🎉 no goals
                                               -/


private lemma aux_IH {R : Type u} {S : Type v} {T : Type w}
  [CommRing R] [CommRing S] [CommRing T] [IsJacobsonRing S] [Algebra R S] [Algebra R T]
  (IH : ∀ (Q : Ideal S), (IsMaximal Q) → RingHom.IsIntegral (algebraMap R (S ⧸ Q)))
  (v : S[X] ≃ₐ[R] T) (P : Ideal T) (hP : P.IsMaximal) :
  RingHom.IsIntegral (algebraMap R (T ⧸ P)) := by
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : IsJacobsonRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
    v : AlgEquiv R (Polynomial S) T
    P : Ideal T
    hP : P.IsMaximal
    ⊢ (algebraMap R (HasQuotient.Quotient T P)).IsIntegral
  -/
  let Q := P.comap v.toAlgHom.toRingHom
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : IsJacobsonRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
    v : AlgEquiv R (Polynomial S) T
    P : Ideal T
    hP : P.IsMaximal
    Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
    ⊢ (algebraMap R (HasQuotient.Quotient T P)).IsIntegral
  -/
  have hw : Ideal.map v Q = P := map_comap_of_surjective v v.surjective P
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : IsJacobsonRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
    v : AlgEquiv R (Polynomial S) T
    P : Ideal T
    hP : P.IsMaximal
    Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
    hw : Eq (Ideal.map v Q) P
    ⊢ (algebraMap R (HasQuotient.Quotient T P)).IsIntegral
  -/
  haveI hQ : IsMaximal Q := comap_isMaximal_of_surjective _ v.surjective
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : IsJacobsonRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
    v : AlgEquiv R (Polynomial S) T
    P : Ideal T
    hP : P.IsMaximal
    Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
    hw : Eq (Ideal.map v Q) P
    hQ : Q.IsMaximal
    ⊢ (algebraMap R (HasQuotient.Quotient T P)).IsIntegral
  -/
  let w : (S[X] ⧸ Q) ≃ₐ[R] (T ⧸ P) := Ideal.quotientEquivAlg Q P v hw.symm
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : IsJacobsonRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
    v : AlgEquiv R (Polynomial S) T
    P : Ideal T
    hP : P.IsMaximal
    Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
    hw : Eq (Ideal.map v Q) P
    hQ : Q.IsMaximal
    w : AlgEquiv R (HasQuotient.Quotient (Polynomial S) Q) (HasQuotient.Quotient T …
    ⊢ (algebraMap R (HasQuotient.Quotient T P)).IsIntegral
  -/
  let Q' := Q.comap (Polynomial.C)
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : IsJacobsonRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
    v : AlgEquiv R (Polynomial S) T
    P : Ideal T
    hP : P.IsMaximal
    Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
    hw : Eq (Ideal.map v Q) P
    hQ : Q.IsMaximal
    w : AlgEquiv R (HasQuotient.Quotient (Polynomial S) Q) (HasQuotient.Quotient T …
    Q' : Ideal S := Ideal.comap Polynomial.C Q
    ⊢ (algebraMap R (HasQuotient.Quotient T P)).IsIntegral
  -/
  let w' : (S ⧸ Q') →ₐ[R] (S[X] ⧸ Q) := Ideal.quotientMapₐ Q (Cₐ R S) le_rfl
  have h_eq : algebraMap R (T ⧸ P) =
    w.toRingEquiv.toRingHom.comp (w'.toRingHom.comp (algebraMap R (S ⧸ Q'))) := by
    ext r
    simp only [AlgEquiv.toAlgHom_eq_coe, AlgHom.toRingHom_eq_coe, AlgEquiv.toRingEquiv_eq_coe,
      RingEquiv.toRingHom_eq_coe, AlgHom.comp_algebraMap_of_tower, coe_comp, coe_coe,
      AlgEquiv.coe_ringEquiv, Function.comp_apply, AlgEquiv.commutes]
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : IsJacobsonRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
    v : AlgEquiv R (Polynomial S) T
    P : Ideal T
    hP : P.IsMaximal
    Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
    hw : Eq (Ideal.map v Q) P
    hQ : Q.IsMaximal
    w : AlgEquiv R (HasQuotient.Quotient (Polynomial S) Q) (HasQuotient.Quotient T …
    Q' : Ideal S := Ideal.comap Polynomial.C Q
    w' : AlgHom R (HasQuotient.Quotient S Q') (HasQuotient.Quotient (Polynomial S) …
    h_eq : Eq (algebraMap R (HasQuotient.Quotient T P)) (w.toRingEquiv.toRingHom.c …
    ⊢ (algebraMap R (HasQuotient.Quotient T P)).IsIntegral
  -/
  rw [h_eq]
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : IsJacobsonRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
    v : AlgEquiv R (Polynomial S) T
    P : Ideal T
    hP : P.IsMaximal
    Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
    hw : Eq (Ideal.map v Q) P
    hQ : Q.IsMaximal
    w : AlgEquiv R (HasQuotient.Quotient (Polynomial S) Q) (HasQuotient.Quotient T …
    Q' : Ideal S := Ideal.comap Polynomial.C Q
    w' : AlgHom R (HasQuotient.Quotient S Q') (HasQuotient.Quotient (Polynomial S) …
    h_eq : Eq (algebraMap R (HasQuotient.Quotient T P)) (w.toRingEquiv.toRingHom.c …
    ⊢ (w.toRingEquiv.toRingHom.comp (w'.comp (algebraMap R (HasQuotient.Quotient S …
  -/
  apply RingHom.IsIntegral.trans
    /-
      case hf
      R : Type u
      S : Type v
      T : Type w
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      inst✝³ : CommRing T
      inst✝² : IsJacobsonRing S
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
      v : AlgEquiv R (Polynomial S) T
      P : Ideal T
      hP : P.IsMaximal
      Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
      hw : Eq (Ideal.map v Q) P
      hQ : Q.IsMaximal
      w : AlgEquiv R (HasQuotient.Quotient (Polynomial S) Q) (HasQuotient.Quotient T …
      Q' : Ideal S := Ideal.comap Polynomial.C Q
      w' : AlgHom R (HasQuotient.Quotient S Q') (HasQuotient.Quotient (Polynomial S) …
      h_eq : Eq (algebraMap R (HasQuotient.Quotient T P)) (w.toRingEquiv.toRingHom.c …
      ⊢ (w'.comp (algebraMap R (HasQuotient.Quotient S Q'))).IsIntegral
    -/
  · apply RingHom.IsIntegral.trans
      /-
        case hf.hf
        R : Type u
        S : Type v
        T : Type w
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing S
        inst✝³ : CommRing T
        inst✝² : IsJacobsonRing S
        inst✝¹ : Algebra R S
        inst✝ : Algebra R T
        IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
        v : AlgEquiv R (Polynomial S) T
        P : Ideal T
        hP : P.IsMaximal
        Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
        hw : Eq (Ideal.map v Q) P
        hQ : Q.IsMaximal
        w : AlgEquiv R (HasQuotient.Quotient (Polynomial S) Q) (HasQuotient.Quotient T …
        Q' : Ideal S := Ideal.comap Polynomial.C Q
        w' : AlgHom R (HasQuotient.Quotient S Q') (HasQuotient.Quotient (Polynomial S) …
        h_eq : Eq (algebraMap R (HasQuotient.Quotient T P)) (w.toRingEquiv.toRingHom.c …
        ⊢ (algebraMap R (HasQuotient.Quotient S Q')).IsIntegral
      -/
    · apply IH
      /-
        case hf.hf.a
        R : Type u
        S : Type v
        T : Type w
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing S
        inst✝³ : CommRing T
        inst✝² : IsJacobsonRing S
        inst✝¹ : Algebra R S
        inst✝ : Algebra R T
        IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
        v : AlgEquiv R (Polynomial S) T
        P : Ideal T
        hP : P.IsMaximal
        Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
        hw : Eq (Ideal.map v Q) P
        hQ : Q.IsMaximal
        w : AlgEquiv R (HasQuotient.Quotient (Polynomial S) Q) (HasQuotient.Quotient T …
        Q' : Ideal S := Ideal.comap Polynomial.C Q
        w' : AlgHom R (HasQuotient.Quotient S Q') (HasQuotient.Quotient (Polynomial S) …
        h_eq : Eq (algebraMap R (HasQuotient.Quotient T P)) (w.toRingEquiv.toRingHom.c …
        ⊢ Q'.IsMaximal
      -/
      apply Polynomial.isMaximal_comap_C_of_isJacobsonRing
      /-
        🎉 no goals
      -/
    · suffices w'.toRingHom = Ideal.quotientMap Q (Polynomial.C) le_rfl by
        rw [this]
        rw [isIntegral_quotientMap_iff _]
        apply Polynomial.quotient_mk_comp_C_isIntegral_of_isJacobsonRing
      /-
        case hf.hg
        R : Type u
        S : Type v
        T : Type w
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing S
        inst✝³ : CommRing T
        inst✝² : IsJacobsonRing S
        inst✝¹ : Algebra R S
        inst✝ : Algebra R T
        IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
        v : AlgEquiv R (Polynomial S) T
        P : Ideal T
        hP : P.IsMaximal
        Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
        hw : Eq (Ideal.map v Q) P
        hQ : Q.IsMaximal
        w : AlgEquiv R (HasQuotient.Quotient (Polynomial S) Q) (HasQuotient.Quotient T …
        Q' : Ideal S := Ideal.comap Polynomial.C Q
        w' : AlgHom R (HasQuotient.Quotient S Q') (HasQuotient.Quotient (Polynomial S) …
        h_eq : Eq (algebraMap R (HasQuotient.Quotient T P)) (w.toRingEquiv.toRingHom.c …
        ⊢ Eq w'.toRingHom (Ideal.quotientMap Q Polynomial.C ⋯)
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case hg
      R : Type u
      S : Type v
      T : Type w
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      inst✝³ : CommRing T
      inst✝² : IsJacobsonRing S
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
      v : AlgEquiv R (Polynomial S) T
      P : Ideal T
      hP : P.IsMaximal
      Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
      hw : Eq (Ideal.map v Q) P
      hQ : Q.IsMaximal
      w : AlgEquiv R (HasQuotient.Quotient (Polynomial S) Q) (HasQuotient.Quotient T …
      Q' : Ideal S := Ideal.comap Polynomial.C Q
      w' : AlgHom R (HasQuotient.Quotient S Q') (HasQuotient.Quotient (Polynomial S) …
      h_eq : Eq (algebraMap R (HasQuotient.Quotient T P)) (w.toRingEquiv.toRingHom.c …
      ⊢ w.toRingEquiv.toRingHom.IsIntegral
    -/
  · apply RingHom.isIntegral_of_surjective
    /-
      case hg.hf
      R : Type u
      S : Type v
      T : Type w
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      inst✝³ : CommRing T
      inst✝² : IsJacobsonRing S
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      IH : ∀ (Q : Ideal S), Q.IsMaximal → (algebraMap R (HasQuotient.Quotient S Q)). …
      v : AlgEquiv R (Polynomial S) T
      P : Ideal T
      hP : P.IsMaximal
      Q : Ideal (Polynomial S) := Ideal.comap (↑v).toRingHom P
      hw : Eq (Ideal.map v Q) P
      hQ : Q.IsMaximal
      w : AlgEquiv R (HasQuotient.Quotient (Polynomial S) Q) (HasQuotient.Quotient T …
      Q' : Ideal S := Ideal.comap Polynomial.C Q
      w' : AlgHom R (HasQuotient.Quotient S Q') (HasQuotient.Quotient (Polynomial S) …
      h_eq : Eq (algebraMap R (HasQuotient.Quotient T P)) (w.toRingEquiv.toRingHom.c …
      ⊢ Function.Surjective ⇑w.toRingEquiv.toRingHom
    -/
    exact w.surjective
    /-
      🎉 no goals
    -/


private theorem quotient_mk_comp_C_isIntegral_of_isJacobsonRing'
    {R : Type*} [CommRing R] [IsJacobsonRing R]
    (P : Ideal (MvPolynomial (Fin n) R)) (hP : P.IsMaximal) :
    RingHom.IsIntegral (algebraMap R (MvPolynomial (Fin n) R ⧸ P)) := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    P : Ideal (MvPolynomial (Fin n) R)
    hP : P.IsMaximal
    ⊢ (algebraMap R (HasQuotient.Quotient (MvPolynomial (Fin n) R) P)).IsIntegral
  -/
  induction' n with n IH
    /-
      case zero
      n : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsJacobsonRing R
      P : Ideal (MvPolynomial (Fin 0) R)
      hP : P.IsMaximal
      ⊢ (algebraMap R (HasQuotient.Quotient (MvPolynomial (Fin 0) R) P)).IsIntegral
    -/
  · apply RingHom.isIntegral_of_surjective
    /-
      case zero.hf
      n : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsJacobsonRing R
      P : Ideal (MvPolynomial (Fin 0) R)
      hP : P.IsMaximal
      ⊢ Function.Surjective ⇑(algebraMap R (HasQuotient.Quotient (MvPolynomial (Fin  …
    -/
    apply Function.Surjective.comp Quotient.mk_surjective
    /-
      case zero.hf
      n : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsJacobsonRing R
      P : Ideal (MvPolynomial (Fin 0) R)
      hP : P.IsMaximal
      ⊢ Function.Surjective ⇑(algebraMap R (MvPolynomial (Fin 0) R))
    -/
    exact C_surjective (Fin 0)
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsJacobsonRing R
      n : Nat
      IH : ∀ (P : Ideal (MvPolynomial (Fin n) R)), P.IsMaximal → (algebraMap R (HasQ …
      P : Ideal (MvPolynomial (Fin (HAdd.hAdd n 1)) R)
      hP : P.IsMaximal
      ⊢ (algebraMap R (HasQuotient.Quotient (MvPolynomial (Fin (HAdd.hAdd n 1)) R) P …
    -/
  · apply aux_IH IH (finSuccEquiv R n).symm P hP
    /-
      🎉 no goals
    -/


theorem quotient_mk_comp_C_isIntegral_of_isJacobsonRing {R : Type*} [CommRing R] [IsJacobsonRing R]
    (P : Ideal (MvPolynomial (Fin n) R)) [hP : P.IsMaximal] :
    RingHom.IsIntegral (RingHom.comp (Ideal.Quotient.mk P) (MvPolynomial.C)) := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    P : Ideal (MvPolynomial (Fin n) R)
    hP : P.IsMaximal
    ⊢ ((Ideal.Quotient.mk P).comp MvPolynomial.C).IsIntegral
  -/
  change RingHom.IsIntegral (algebraMap R (MvPolynomial (Fin n) R ⧸ P))
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    P : Ideal (MvPolynomial (Fin n) R)
    hP : P.IsMaximal
    ⊢ (algebraMap R (HasQuotient.Quotient (MvPolynomial (Fin n) R) P)).IsIntegral
  -/
  apply quotient_mk_comp_C_isIntegral_of_isJacobsonRing'
  /-
    case hP
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    P : Ideal (MvPolynomial (Fin n) R)
    hP : P.IsMaximal
    ⊢ P.IsMaximal
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem comp_C_integral_of_surjective_of_isJacobsonRing {R : Type*} [CommRing R] [IsJacobsonRing R]
    {σ : Type*} [Finite σ] {S : Type*} [Field S] (f : MvPolynomial σ R →+* S)
    (hf : Function.Surjective ↑f) : (f.comp C).IsIntegral := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsJacobsonRing R
    σ : Type u_2
    inst✝¹ : Finite σ
    S : Type u_3
    inst✝ : Field S
    f : RingHom (MvPolynomial σ R) S
    hf : Function.Surjective ⇑f
    ⊢ (f.comp MvPolynomial.C).IsIntegral
  -/
  cases nonempty_fintype σ
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsJacobsonRing R
    σ : Type u_2
    inst✝¹ : Finite σ
    S : Type u_3
    inst✝ : Field S
    f : RingHom (MvPolynomial σ R) S
    hf : Function.Surjective ⇑f
    val✝ : Fintype σ
    ⊢ (f.comp MvPolynomial.C).IsIntegral
  -/
  have e := (Fintype.equivFin σ).symm
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsJacobsonRing R
    σ : Type u_2
    inst✝¹ : Finite σ
    S : Type u_3
    inst✝ : Field S
    f : RingHom (MvPolynomial σ R) S
    hf : Function.Surjective ⇑f
    val✝ : Fintype σ
    e : Equiv (Fin (Fintype.card σ)) σ
    ⊢ (f.comp MvPolynomial.C).IsIntegral
  -/
  let f' : MvPolynomial (Fin _) R →+* S := f.comp (renameEquiv R e).toRingEquiv.toRingHom
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsJacobsonRing R
    σ : Type u_2
    inst✝¹ : Finite σ
    S : Type u_3
    inst✝ : Field S
    f : RingHom (MvPolynomial σ R) S
    hf : Function.Surjective ⇑f
    val✝ : Fintype σ
    e : Equiv (Fin (Fintype.card σ)) σ
    f' : RingHom (MvPolynomial (Fin (Fintype.card σ)) R) S := f.comp (MvPolynomial …
    ⊢ (f.comp MvPolynomial.C).IsIntegral
  -/
  have hf' := Function.Surjective.comp hf (renameEquiv R e).surjective
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsJacobsonRing R
    σ : Type u_2
    inst✝¹ : Finite σ
    S : Type u_3
    inst✝ : Field S
    f : RingHom (MvPolynomial σ R) S
    hf : Function.Surjective ⇑f
    val✝ : Fintype σ
    e : Equiv (Fin (Fintype.card σ)) σ
    f' : RingHom (MvPolynomial (Fin (Fintype.card σ)) R) S := f.comp (MvPolynomial …
    hf' : Function.Surjective (Function.comp ⇑f ⇑(MvPolynomial.renameEquiv R e))
    ⊢ (f.comp MvPolynomial.C).IsIntegral
  -/
  change Function.Surjective ↑f' at hf'
  have : (f'.comp C).IsIntegral := by
    haveI : f'.ker.IsMaximal := ker_isMaximal_of_surjective f' hf'
    let g : MvPolynomial _ R ⧸ (RingHom.ker f') →+* S :=
      Ideal.Quotient.lift (RingHom.ker f') f' fun _ h => h
    have hfg : g.comp (Ideal.Quotient.mk (RingHom.ker f')) = f' :=
      ringHom_ext (fun r => rfl) fun i => rfl
    rw [← hfg, RingHom.comp_assoc]
    refine (quotient_mk_comp_C_isIntegral_of_isJacobsonRing (RingHom.ker f')).trans _ g
      (g.isIntegral_of_surjective ?_)
    rw [← hfg] at hf'
    norm_num at hf'
    exact Function.Surjective.of_comp hf'
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsJacobsonRing R
    σ : Type u_2
    inst✝¹ : Finite σ
    S : Type u_3
    inst✝ : Field S
    f : RingHom (MvPolynomial σ R) S
    hf : Function.Surjective ⇑f
    val✝ : Fintype σ
    e : Equiv (Fin (Fintype.card σ)) σ
    f' : RingHom (MvPolynomial (Fin (Fintype.card σ)) R) S := f.comp (MvPolynomial …
    hf' : Function.Surjective ⇑f'
    this : (f'.comp MvPolynomial.C).IsIntegral
    ⊢ (f.comp MvPolynomial.C).IsIntegral
  -/
  rw [RingHom.comp_assoc] at this
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsJacobsonRing R
    σ : Type u_2
    inst✝¹ : Finite σ
    S : Type u_3
    inst✝ : Field S
    f : RingHom (MvPolynomial σ R) S
    hf : Function.Surjective ⇑f
    val✝ : Fintype σ
    e : Equiv (Fin (Fintype.card σ)) σ
    f' : RingHom (MvPolynomial (Fin (Fintype.card σ)) R) S := f.comp (MvPolynomial …
    hf' : Function.Surjective ⇑f'
    this : (f.comp ((MvPolynomial.renameEquiv R e).toRingEquiv.toRingHom.comp MvPo …
    ⊢ (f.comp MvPolynomial.C).IsIntegral
  -/
  convert this
  /-
    case h.e'_5.h.e'_8
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsJacobsonRing R
    σ : Type u_2
    inst✝¹ : Finite σ
    S : Type u_3
    inst✝ : Field S
    f : RingHom (MvPolynomial σ R) S
    hf : Function.Surjective ⇑f
    val✝ : Fintype σ
    e : Equiv (Fin (Fintype.card σ)) σ
    f' : RingHom (MvPolynomial (Fin (Fintype.card σ)) R) S := f.comp (MvPolynomial …
    hf' : Function.Surjective ⇑f'
    this : (f.comp ((MvPolynomial.renameEquiv R e).toRingEquiv.toRingHom.comp MvPo …
    ⊢ Eq MvPolynomial.C ((MvPolynomial.renameEquiv R e).toRingEquiv.toRingHom.comp …
  -/
  refine RingHom.ext fun x => ?_
  /-
    case h.e'_5.h.e'_8
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsJacobsonRing R
    σ : Type u_2
    inst✝¹ : Finite σ
    S : Type u_3
    inst✝ : Field S
    f : RingHom (MvPolynomial σ R) S
    hf : Function.Surjective ⇑f
    val✝ : Fintype σ
    e : Equiv (Fin (Fintype.card σ)) σ
    f' : RingHom (MvPolynomial (Fin (Fintype.card σ)) R) S := f.comp (MvPolynomial …
    hf' : Function.Surjective ⇑f'
    this : (f.comp ((MvPolynomial.renameEquiv R e).toRingEquiv.toRingHom.comp MvPo …
    x : R
    ⊢ Eq (MvPolynomial.C x) (((MvPolynomial.renameEquiv R e).toRingEquiv.toRingHom …
  -/
  exact ((renameEquiv R e).commutes' x).symm
  /-
    🎉 no goals
  -/


lemma isJacobsonRing_of_finiteType {A B : Type*} [CommRing A] [CommRing B]
    [Algebra A B] [IsJacobsonRing A] [Algebra.FiniteType A B] : IsJacobsonRing B := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    inst✝¹ : IsJacobsonRing A
    inst✝ : Algebra.FiniteType A B
    ⊢ IsJacobsonRing B
  -/
  obtain ⟨ι, hι, f, hf⟩ := Algebra.FiniteType.iff_quotient_mvPolynomial'.mp ‹_›
  /-
    case intro.intro.intro
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    inst✝¹ : IsJacobsonRing A
    inst✝ : Algebra.FiniteType A B
    ι : Type u_2
    hι : Fintype ι
    f : AlgHom A (MvPolynomial ι A) B
    hf : Function.Surjective ⇑f
    ⊢ IsJacobsonRing B
  -/
  exact isJacobsonRing_of_surjective ⟨f.toRingHom, hf⟩
  /-
    🎉 no goals
  -/


lemma RingHom.FiniteType.isJacobsonRing {A B : Type*} [CommRing A] [CommRing B]
    {f : A →+* B} [IsJacobsonRing A] (H : f.FiniteType) : IsJacobsonRing B :=
  @isJacobsonRing_of_finiteType A B _ _ f.toAlgebra _ H


lemma finite_of_finite_type_of_isJacobsonRing (R S : Type*) [CommRing R] [Field S]
    [Algebra R S] [IsJacobsonRing R] [Algebra.FiniteType R S] :
    Module.Finite R S := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Field S
    inst✝² : Algebra R S
    inst✝¹ : IsJacobsonRing R
    inst✝ : Algebra.FiniteType R S
    ⊢ Module.Finite R S
  -/
  obtain ⟨ι, hι, f, hf⟩ := Algebra.FiniteType.iff_quotient_mvPolynomial'.mp ‹_›
  have : (algebraMap R S).IsIntegral := by
    rw [← f.comp_algebraMap]
    #adaptation_note
    /--
    After https://github.com/leanprover/lean4/pull/6024
    we needed to write `f.toRingHom` instead of just `f`, to avoid unification issues.
    -/
    exact MvPolynomial.comp_C_integral_of_surjective_of_isJacobsonRing f.toRingHom hf
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Field S
    inst✝² : Algebra R S
    inst✝¹ : IsJacobsonRing R
    inst✝ : Algebra.FiniteType R S
    ι : Type u_2
    hι : Fintype ι
    f : AlgHom R (MvPolynomial ι R) S
    hf : Function.Surjective ⇑f
    this : (algebraMap R S).IsIntegral
    ⊢ Module.Finite R S
  -/
  have : Algebra.IsIntegral R S := Algebra.isIntegral_def.mpr this
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Field S
    inst✝² : Algebra R S
    inst✝¹ : IsJacobsonRing R
    inst✝ : Algebra.FiniteType R S
    ι : Type u_2
    hι : Fintype ι
    f : AlgHom R (MvPolynomial ι R) S
    hf : Function.Surjective ⇑f
    this✝ : (algebraMap R S).IsIntegral
    this : Algebra.IsIntegral R S
    ⊢ Module.Finite R S
  -/
  exact Algebra.IsIntegral.finite
  /-
    🎉 no goals
  -/


/--
If `f : R →+* S` is a ring homomorphism from a jacobson ring to a field,
then it is finite if and only if it is finite type.
-/
lemma RingHom.finite_iff_finiteType_of_isJacobsonRing
    {R S : Type*} [CommRing R] [IsJacobsonRing R] [Field S]
    {f : R →+* S} : f.Finite ↔ f.FiniteType :=
  ⟨RingHom.FiniteType.of_finite,
       /-
         R : Type u_1
         S : Type u_2
         inst✝² : CommRing R
         inst✝¹ : IsJacobsonRing R
         inst✝ : Field S
         f : RingHom R S
         ⊢ f.FiniteType → f.Finite
       -/
    by intro; algebraize [f]; exact finite_of_finite_type_of_isJacobsonRing R S⟩
                              /-
                                🎉 no goals
                              -/


@[deprecated (since := "2024-10-27")]
alias IsJacobson := IsJacobsonRing

@[deprecated (since := "2024-10-27")]
alias isJacobson_iff := isJacobsonRing_iff

@[deprecated (since := "2024-10-27")]
alias IsJacobson.out := IsJacobsonRing.out

@[deprecated (since := "2024-10-27")]
alias isJacobson_iff_prime_eq := isJacobsonRing_iff_prime_eq

@[deprecated (since := "2024-10-27")]
alias isJacobson_iff_sInf_maximal := isJacobsonRing_iff_sInf_maximal

@[deprecated (since := "2024-10-27")]
alias isJacobson_iff_sInf_maximal' := isJacobsonRing_iff_sInf_maximal'

@[deprecated (since := "2024-10-27")]
alias isJacobson_of_surjective := isJacobsonRing_of_surjective

@[deprecated (since := "2024-10-27")]
alias isJacobson_iso := isJacobsonRing_iso

@[deprecated (since := "2024-10-27")]
alias isJacobson_of_isIntegral := isJacobsonRing_of_isIntegral

@[deprecated (since := "2024-10-27")]
alias isJacobson_of_isIntegral' := isJacobsonRing_of_isIntegral'

@[deprecated (since := "2024-10-27")]
alias isMaximal_iff_isMaximal_disjoint := IsLocalization.isMaximal_iff_isMaximal_disjoint

@[deprecated (since := "2024-10-27")]
alias isMaximal_of_isMaximal_disjoint := IsLocalization.isMaximal_of_isMaximal_disjoint

@[deprecated (since := "2024-10-27")]
alias isJacobson_localization := isJacobsonRing_localization


@[deprecated (since := "2024-10-27")]
alias isIntegral_isLocalization_polynomial_quotient := isIntegral_isLocalization_polynomial_quotient

@[deprecated (since := "2024-10-27")]
alias jacobson_bot_of_integral_localization := jacobson_bot_of_integral_localization

@[deprecated (since := "2024-10-27")]
alias isJacobson_polynomial_of_isJacobson := isJacobsonRing_polynomial_of_isJacobsonRing

@[deprecated (since := "2024-10-27")]
alias isJacobson_polynomial_iff_isJacobson := isJacobsonRing_polynomial_iff_isJacobsonRing

@[deprecated (since := "2024-10-27")]
alias isMaximal_comap_C_of_isMaximal := isMaximal_comap_C_of_isMaximal

@[deprecated (since := "2024-10-27")]
alias quotient_mk_comp_C_isIntegral_of_jacobson :=
  Polynomial.quotient_mk_comp_C_isIntegral_of_isJacobsonRing

@[deprecated (since := "2024-10-27")]
alias isMaximal_comap_C_of_isJacobson := isMaximal_comap_C_of_isJacobsonRing

@[deprecated (since := "2024-10-27")]
alias comp_C_integral_of_surjective_of_jacobson :=
  Polynomial.comp_C_integral_of_surjective_of_isJacobsonRing


@[deprecated (since := "2024-10-27")]
alias MvPolynomial.isJacobson_MvPolynomial_fin := isJacobsonRing_MvPolynomial_fin

@[deprecated (since := "2024-10-27")]
alias MvPolynomial.quotient_mk_comp_C_isIntegral_of_jacobson :=
  MvPolynomial.quotient_mk_comp_C_isIntegral_of_isJacobsonRing

@[deprecated (since := "2024-10-27")]
alias MvPolynomial.comp_C_integral_of_surjective_of_jacobson :=
  MvPolynomial.comp_C_integral_of_surjective_of_isJacobsonRing


