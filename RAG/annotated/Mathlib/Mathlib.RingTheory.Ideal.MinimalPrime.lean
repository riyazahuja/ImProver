/-- `I.minimalPrimes` is the set of ideals that are minimal primes over `I`. -/
protected def Ideal.minimalPrimes : Set (Ideal R) :=
  {p | Minimal (fun q ↦ q.IsPrime ∧ I ≤ q) p}


variable (R) in
/-- `minimalPrimes R` is the set of minimal primes of `R`.
This is defined as `Ideal.minimalPrimes ⊥`. -/
def minimalPrimes : Set (Ideal R) :=
  Ideal.minimalPrimes ⊥


lemma minimalPrimes_eq_minimals : minimalPrimes R = {x | Minimal Ideal.IsPrime x} :=
                        /-
                          R : Type u_1
                          inst✝ : CommSemiring R
                          ⊢ Eq (fun q => And q.IsPrime (LE.le Bot.bot q)) Ideal.IsPrime
                        -/
  congr_arg Minimal (by simp)
                        /-
                          🎉 no goals
                        -/


theorem Ideal.exists_minimalPrimes_le [J.IsPrime] (e : I ≤ J) : ∃ p ∈ I.minimalPrimes, p ≤ J := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    I J : Ideal R
    inst✝ : J.IsPrime
    e : LE.le I J
    ⊢ Exists fun p => And (Membership.mem I.minimalPrimes p) (LE.le p J)
  -/
  set S := { p : (Ideal R)ᵒᵈ | Ideal.IsPrime p ∧ I ≤ OrderDual.ofDual p }
  suffices h : ∃ m, OrderDual.toDual J ≤ m ∧ Maximal (· ∈ S) m by
    obtain ⟨p, hJp, hp⟩ := h
    exact ⟨p, ⟨hp.prop, fun q hq hle ↦ hp.le_of_ge hq hle⟩, hJp⟩
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    I J : Ideal R
    inst✝ : J.IsPrime
    e : LE.le I J
    S : Set (OrderDual (Ideal R)) := setOf fun p => And (Ideal.IsPrime p) (LE.le I …
    ⊢ Exists fun m => And (LE.le (OrderDual.toDual J) m) (Maximal (fun x => Member …
  -/
  apply zorn_le_nonempty₀
  /-
    case ih
    R : Type u_1
    inst✝¹ : CommSemiring R
    I J : Ideal R
    inst✝ : J.IsPrime
    e : LE.le I J
    S : Set (OrderDual (Ideal R)) := setOf fun p => And (Ideal.IsPrime p) (LE.le I …
    ⊢ ∀ (c : Set (OrderDual (Ideal R))), HasSubset.Subset c S → IsChain (fun x1 x2 …
  -/
  swap
    /-
      case hxs
      R : Type u_1
      inst✝¹ : CommSemiring R
      I J : Ideal R
      inst✝ : J.IsPrime
      e : LE.le I J
      S : Set (OrderDual (Ideal R)) := setOf fun p => And (Ideal.IsPrime p) (LE.le I …
      ⊢ Membership.mem S (OrderDual.toDual J)
    -/
  · refine ⟨show J.IsPrime by infer_instance, e⟩
    /-
      🎉 no goals
    -/
  /-
    case ih
    R : Type u_1
    inst✝¹ : CommSemiring R
    I J : Ideal R
    inst✝ : J.IsPrime
    e : LE.le I J
    S : Set (OrderDual (Ideal R)) := setOf fun p => And (Ideal.IsPrime p) (LE.le I …
    ⊢ ∀ (c : Set (OrderDual (Ideal R))), HasSubset.Subset c S → IsChain (fun x1 x2 …
  -/
  rintro (c : Set (Ideal R)) hc hc' J' hJ'
  refine
    ⟨OrderDual.toDual (sInf c),
      ⟨Ideal.sInf_isPrime_of_isChain ⟨J', hJ'⟩ hc'.symm fun x hx => (hc hx).1, ?_⟩, ?_⟩
    /-
      case ih.refine_1
      R : Type u_1
      inst✝¹ : CommSemiring R
      I J : Ideal R
      inst✝ : J.IsPrime
      e : LE.le I J
      S : Set (OrderDual (Ideal R)) := setOf fun p => And (Ideal.IsPrime p) (LE.le I …
      c : Set (Ideal R)
      hc : HasSubset.Subset c S
      hc' : IsChain (fun x1 x2 => LE.le x1 x2) c
      J' : OrderDual (Ideal R)
      hJ' : Membership.mem c J'
      ⊢ LE.le I (OrderDual.ofDual (OrderDual.toDual (InfSet.sInf c)))
    -/
  · rw [OrderDual.ofDual_toDual, le_sInf_iff]
    /-
      case ih.refine_1
      R : Type u_1
      inst✝¹ : CommSemiring R
      I J : Ideal R
      inst✝ : J.IsPrime
      e : LE.le I J
      S : Set (OrderDual (Ideal R)) := setOf fun p => And (Ideal.IsPrime p) (LE.le I …
      c : Set (Ideal R)
      hc : HasSubset.Subset c S
      hc' : IsChain (fun x1 x2 => LE.le x1 x2) c
      J' : OrderDual (Ideal R)
      hJ' : Membership.mem c J'
      ⊢ ∀ (b : Ideal R), Membership.mem c b → LE.le I b
    -/
    exact fun _ hx => (hc hx).2
    /-
      🎉 no goals
    -/
    /-
      case ih.refine_2
      R : Type u_1
      inst✝¹ : CommSemiring R
      I J : Ideal R
      inst✝ : J.IsPrime
      e : LE.le I J
      S : Set (OrderDual (Ideal R)) := setOf fun p => And (Ideal.IsPrime p) (LE.le I …
      c : Set (Ideal R)
      hc : HasSubset.Subset c S
      hc' : IsChain (fun x1 x2 => LE.le x1 x2) c
      J' : OrderDual (Ideal R)
      hJ' : Membership.mem c J'
      ⊢ ∀ (z : OrderDual (Ideal R)), Membership.mem c z → LE.le z (OrderDual.toDual  …
    -/
  · rintro z hz
    /-
      case ih.refine_2
      R : Type u_1
      inst✝¹ : CommSemiring R
      I J : Ideal R
      inst✝ : J.IsPrime
      e : LE.le I J
      S : Set (OrderDual (Ideal R)) := setOf fun p => And (Ideal.IsPrime p) (LE.le I …
      c : Set (Ideal R)
      hc : HasSubset.Subset c S
      hc' : IsChain (fun x1 x2 => LE.le x1 x2) c
      J' : OrderDual (Ideal R)
      hJ' : Membership.mem c J'
      z : OrderDual (Ideal R)
      hz : Membership.mem c z
      ⊢ LE.le z (OrderDual.toDual (InfSet.sInf c))
    -/
    rw [OrderDual.le_toDual]
    /-
      case ih.refine_2
      R : Type u_1
      inst✝¹ : CommSemiring R
      I J : Ideal R
      inst✝ : J.IsPrime
      e : LE.le I J
      S : Set (OrderDual (Ideal R)) := setOf fun p => And (Ideal.IsPrime p) (LE.le I …
      c : Set (Ideal R)
      hc : HasSubset.Subset c S
      hc' : IsChain (fun x1 x2 => LE.le x1 x2) c
      J' : OrderDual (Ideal R)
      hJ' : Membership.mem c J'
      z : OrderDual (Ideal R)
      hz : Membership.mem c z
      ⊢ LE.le (InfSet.sInf c) (OrderDual.ofDual z)
    -/
    exact sInf_le hz
    /-
      🎉 no goals
    -/


@[simp]
theorem Ideal.radical_minimalPrimes : I.radical.minimalPrimes = I.minimalPrimes := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ Eq I.radical.minimalPrimes I.minimalPrimes
  -/
  rw [Ideal.minimalPrimes, Ideal.minimalPrimes]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ Eq (setOf fun p => Minimal (fun q => And q.IsPrime (LE.le I.radical q)) p) ( …
  -/
  ext p
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    I p : Ideal R
    ⊢ Iff (Membership.mem (setOf fun p => Minimal (fun q => And q.IsPrime (LE.le I …
  -/
  refine ⟨?_, ?_⟩ <;> rintro ⟨⟨a, ha⟩, b⟩
    /-
      case h.refine_1.intro.intro
      R : Type u_1
      inst✝ : CommSemiring R
      I p : Ideal R
      b : ∀ ⦃y : Ideal R⦄, (fun q => And q.IsPrime (LE.le I.radical q)) y → LE.le y  …
      a : p.IsPrime
      ha : LE.le I.radical p
      ⊢ Membership.mem (setOf fun p => Minimal (fun q => And q.IsPrime (LE.le I q))  …
    -/
  · refine ⟨⟨a, a.radical_le_iff.1 ha⟩, ?_⟩
    /-
      case h.refine_1.intro.intro
      R : Type u_1
      inst✝ : CommSemiring R
      I p : Ideal R
      b : ∀ ⦃y : Ideal R⦄, (fun q => And q.IsPrime (LE.le I.radical q)) y → LE.le y  …
      a : p.IsPrime
      ha : LE.le I.radical p
      ⊢ ∀ ⦃y : Ideal R⦄, (fun q => And q.IsPrime (LE.le I q)) y → LE.le y p → LE.le  …
    -/
    simp only [Set.mem_setOf_eq, and_imp] at *
    /-
      case h.refine_1.intro.intro
      R : Type u_1
      inst✝ : CommSemiring R
      I p : Ideal R
      a : p.IsPrime
      ha : LE.le I.radical p
      b : ∀ ⦃y : Ideal R⦄, y.IsPrime → LE.le I.radical y → LE.le y p → LE.le p y
      ⊢ ∀ ⦃y : Ideal R⦄, y.IsPrime → LE.le I y → LE.le y p → LE.le p y
    -/
    exact fun _ h2 h3 h4 => b h2 (h2.radical_le_iff.2 h3) h4
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2.intro.intro
      R : Type u_1
      inst✝ : CommSemiring R
      I p : Ideal R
      b : ∀ ⦃y : Ideal R⦄, (fun q => And q.IsPrime (LE.le I q)) y → LE.le y p → LE.l …
      a : p.IsPrime
      ha : LE.le I p
      ⊢ Membership.mem (setOf fun p => Minimal (fun q => And q.IsPrime (LE.le I.radi …
    -/
  · refine ⟨⟨a, a.radical_le_iff.2 ha⟩, ?_⟩
    /-
      case h.refine_2.intro.intro
      R : Type u_1
      inst✝ : CommSemiring R
      I p : Ideal R
      b : ∀ ⦃y : Ideal R⦄, (fun q => And q.IsPrime (LE.le I q)) y → LE.le y p → LE.l …
      a : p.IsPrime
      ha : LE.le I p
      ⊢ ∀ ⦃y : Ideal R⦄, (fun q => And q.IsPrime (LE.le I.radical q)) y → LE.le y p  …
    -/
    simp only [Set.mem_setOf_eq, and_imp] at *
    /-
      case h.refine_2.intro.intro
      R : Type u_1
      inst✝ : CommSemiring R
      I p : Ideal R
      a : p.IsPrime
      ha : LE.le I p
      b : ∀ ⦃y : Ideal R⦄, y.IsPrime → LE.le I y → LE.le y p → LE.le p y
      ⊢ ∀ ⦃y : Ideal R⦄, y.IsPrime → LE.le I.radical y → LE.le y p → LE.le p y
    -/
    exact fun _ h2 h3 h4 => b h2 (h2.radical_le_iff.1 h3) h4
    /-
      🎉 no goals
    -/


@[simp]
theorem Ideal.sInf_minimalPrimes : sInf I.minimalPrimes = I.radical := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ Eq (InfSet.sInf I.minimalPrimes) I.radical
  -/
  rw [I.radical_eq_sInf]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ Eq (InfSet.sInf I.minimalPrimes) (InfSet.sInf (setOf fun J => And (LE.le I J …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      inst✝ : CommSemiring R
      I : Ideal R
      ⊢ LE.le (InfSet.sInf I.minimalPrimes) (InfSet.sInf (setOf fun J => And (LE.le  …
    -/
  · intro x hx
    /-
      case a
      R : Type u_1
      inst✝ : CommSemiring R
      I : Ideal R
      x : R
      hx : Membership.mem (InfSet.sInf I.minimalPrimes) x
      ⊢ Membership.mem (InfSet.sInf (setOf fun J => And (LE.le I J) J.IsPrime)) x
    -/
    rw [Ideal.mem_sInf] at hx ⊢
    /-
      case a
      R : Type u_1
      inst✝ : CommSemiring R
      I : Ideal R
      x : R
      hx : ∀ ⦃I_1 : Ideal R⦄, Membership.mem I.minimalPrimes I_1 → Membership.mem I_ …
      ⊢ ∀ ⦃I_1 : Ideal R⦄, Membership.mem (setOf fun J => And (LE.le I J) J.IsPrime) …
    -/
    rintro J ⟨e, hJ⟩
    /-
      case a.intro
      R : Type u_1
      inst✝ : CommSemiring R
      I : Ideal R
      x : R
      hx : ∀ ⦃I_1 : Ideal R⦄, Membership.mem I.minimalPrimes I_1 → Membership.mem I_ …
      J : Ideal R
      e : LE.le I J
      hJ : J.IsPrime
      ⊢ Membership.mem J x
    -/
    obtain ⟨p, hp, hp'⟩ := Ideal.exists_minimalPrimes_le e
    /-
      case a.intro.intro.intro
      R : Type u_1
      inst✝ : CommSemiring R
      I : Ideal R
      x : R
      hx : ∀ ⦃I_1 : Ideal R⦄, Membership.mem I.minimalPrimes I_1 → Membership.mem I_ …
      J : Ideal R
      e : LE.le I J
      hJ : J.IsPrime
      p : Ideal R
      hp : Membership.mem I.minimalPrimes p
      hp' : LE.le p J
      ⊢ Membership.mem J x
    -/
    exact hp' (hx hp)
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_1
      inst✝ : CommSemiring R
      I : Ideal R
      ⊢ LE.le (InfSet.sInf (setOf fun J => And (LE.le I J) J.IsPrime)) (InfSet.sInf  …
    -/
  · apply sInf_le_sInf _
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      I : Ideal R
      ⊢ HasSubset.Subset I.minimalPrimes (setOf fun J => And (LE.le I J) J.IsPrime)
    -/
    intro I hI
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      I✝ I : Ideal R
      hI : Membership.mem I✝.minimalPrimes I
      ⊢ Membership.mem (setOf fun J => And (LE.le I✝ J) J.IsPrime) I
    -/
    exact hI.1.symm
    /-
      🎉 no goals
    -/


theorem Ideal.exists_comap_eq_of_mem_minimalPrimes_of_injective {f : R →+* S}
    (hf : Function.Injective f) (p) (H : p ∈ minimalPrimes R) :
    ∃ p' : Ideal S, p'.IsPrime ∧ p'.comap f = p := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Ideal R
    H : Membership.mem (minimalPrimes R) p
    ⊢ Exists fun p' => And p'.IsPrime (Eq (Ideal.comap f p') p)
  -/
  have := H.1.1
  have : Nontrivial (Localization (Submonoid.map f p.primeCompl)) := by
    refine ⟨⟨1, 0, ?_⟩⟩
    convert (IsLocalization.map_injective_of_injective p.primeCompl (Localization.AtPrime p)
        (Localization <| p.primeCompl.map f) hf).ne one_ne_zero
    · rw [map_one]
    · rw [map_zero]
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Ideal R
    H : Membership.mem (minimalPrimes R) p
    this✝ : p.IsPrime
    this : Nontrivial (Localization (Submonoid.map f p.primeCompl))
    ⊢ Exists fun p' => And p'.IsPrime (Eq (Ideal.comap f p') p)
  -/
  obtain ⟨M, hM⟩ := Ideal.exists_maximal (Localization (Submonoid.map f p.primeCompl))
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Ideal R
    H : Membership.mem (minimalPrimes R) p
    this✝ : p.IsPrime
    this : Nontrivial (Localization (Submonoid.map f p.primeCompl))
    M : Ideal (Localization (Submonoid.map f p.primeCompl))
    hM : M.IsMaximal
    ⊢ Exists fun p' => And p'.IsPrime (Eq (Ideal.comap f p') p)
  -/
  refine ⟨M.comap (algebraMap S <| Localization (Submonoid.map f p.primeCompl)), inferInstance, ?_⟩
  rw [Ideal.comap_comap, ← @IsLocalization.map_comp _ _ _ _ _ _ _ _ Localization.isLocalization
      _ _ _ _ _ Localization.isLocalization p.primeCompl.le_comap_map,
    ← Ideal.comap_comap]
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Ideal R
    H : Membership.mem (minimalPrimes R) p
    this✝ : p.IsPrime
    this : Nontrivial (Localization (Submonoid.map f p.primeCompl))
    M : Ideal (Localization (Submonoid.map f p.primeCompl))
    hM : M.IsMaximal
    ⊢ Eq (Ideal.comap (algebraMap R (Localization p.primeCompl)) (Ideal.comap (IsL …
  -/
  suffices _ ≤ p by exact this.antisymm (H.2 ⟨inferInstance, bot_le⟩ this)
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Ideal R
    H : Membership.mem (minimalPrimes R) p
    this✝ : p.IsPrime
    this : Nontrivial (Localization (Submonoid.map f p.primeCompl))
    M : Ideal (Localization (Submonoid.map f p.primeCompl))
    hM : M.IsMaximal
    ⊢ LE.le (Ideal.comap (algebraMap R (Localization p.primeCompl)) (Ideal.comap ( …
  -/
  intro x hx
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Ideal R
    H : Membership.mem (minimalPrimes R) p
    this✝ : p.IsPrime
    this : Nontrivial (Localization (Submonoid.map f p.primeCompl))
    M : Ideal (Localization (Submonoid.map f p.primeCompl))
    hM : M.IsMaximal
    x : R
    hx : Membership.mem (Ideal.comap (algebraMap R (Localization p.primeCompl)) (I …
    ⊢ Membership.mem p x
  -/
  by_contra h
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Ideal R
    H : Membership.mem (minimalPrimes R) p
    this✝ : p.IsPrime
    this : Nontrivial (Localization (Submonoid.map f p.primeCompl))
    M : Ideal (Localization (Submonoid.map f p.primeCompl))
    hM : M.IsMaximal
    x : R
    hx : Membership.mem (Ideal.comap (algebraMap R (Localization p.primeCompl)) (I …
    h : Not (Membership.mem p x)
    ⊢ False
  -/
  apply hM.ne_top
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Ideal R
    H : Membership.mem (minimalPrimes R) p
    this✝ : p.IsPrime
    this : Nontrivial (Localization (Submonoid.map f p.primeCompl))
    M : Ideal (Localization (Submonoid.map f p.primeCompl))
    hM : M.IsMaximal
    x : R
    hx : Membership.mem (Ideal.comap (algebraMap R (Localization p.primeCompl)) (I …
    h : Not (Membership.mem p x)
    ⊢ Eq M Top.top
  -/
  apply M.eq_top_of_isUnit_mem hx
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Ideal R
    H : Membership.mem (minimalPrimes R) p
    this✝ : p.IsPrime
    this : Nontrivial (Localization (Submonoid.map f p.primeCompl))
    M : Ideal (Localization (Submonoid.map f p.primeCompl))
    hM : M.IsMaximal
    x : R
    hx : Membership.mem (Ideal.comap (algebraMap R (Localization p.primeCompl)) (I …
    h : Not (Membership.mem p x)
    ⊢ IsUnit ((IsLocalization.map (Localization (Submonoid.map f p.primeCompl)) f  …
  -/
  apply IsUnit.map
  /-
    case intro.h
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Ideal R
    H : Membership.mem (minimalPrimes R) p
    this✝ : p.IsPrime
    this : Nontrivial (Localization (Submonoid.map f p.primeCompl))
    M : Ideal (Localization (Submonoid.map f p.primeCompl))
    hM : M.IsMaximal
    x : R
    hx : Membership.mem (Ideal.comap (algebraMap R (Localization p.primeCompl)) (I …
    h : Not (Membership.mem p x)
    ⊢ IsUnit ((algebraMap R (Localization p.primeCompl)) x)
  -/
  apply IsLocalization.map_units _ (show p.primeCompl from ⟨x, h⟩)
  /-
    🎉 no goals
  -/


theorem Ideal.exists_comap_eq_of_mem_minimalPrimes {I : Ideal S} (f : R →+* S) (p)
    (H : p ∈ (I.comap f).minimalPrimes) : ∃ p' : Ideal S, p'.IsPrime ∧ I ≤ p' ∧ p'.comap f = p := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    I : Ideal S
    f : RingHom R S
    p : Ideal R
    H : Membership.mem (Ideal.comap f I).minimalPrimes p
    ⊢ Exists fun p' => And p'.IsPrime (And (LE.le I p') (Eq (Ideal.comap f p') p))
  -/
  have := H.1.1
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    I : Ideal S
    f : RingHom R S
    p : Ideal R
    H : Membership.mem (Ideal.comap f I).minimalPrimes p
    this : p.IsPrime
    ⊢ Exists fun p' => And p'.IsPrime (And (LE.le I p') (Eq (Ideal.comap f p') p))
  -/
  let f' := (Ideal.Quotient.mk I).comp f
  have e : RingHom.ker f' = I.comap f := by
    ext1
    exact Submodule.Quotient.mk_eq_zero _
  have : RingHom.ker (Ideal.Quotient.mk <| RingHom.ker f') ≤ p := by
    rw [Ideal.mk_ker, e]
    exact H.1.2
  suffices _ by
    have ⟨p', hp₁, hp₂⟩ := Ideal.exists_comap_eq_of_mem_minimalPrimes_of_injective
      (RingHom.kerLift_injective f') (p.map <| Ideal.Quotient.mk <| RingHom.ker f') this
    refine ⟨p'.comap <| Ideal.Quotient.mk I, Ideal.IsPrime.comap _, ?_, ?_⟩
    · exact Ideal.mk_ker.symm.trans_le (Ideal.comap_mono bot_le)
    · convert congr_arg (Ideal.comap <| Ideal.Quotient.mk <| RingHom.ker f') hp₂
      rwa [Ideal.comap_map_of_surjective (Ideal.Quotient.mk <| RingHom.ker f')
        Ideal.Quotient.mk_surjective, eq_comm, sup_eq_left]
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    I : Ideal S
    f : RingHom R S
    p : Ideal R
    H : Membership.mem (Ideal.comap f I).minimalPrimes p
    this✝ : p.IsPrime
    f' : RingHom R (HasQuotient.Quotient S I) := (Ideal.Quotient.mk I).comp f
    e : Eq (RingHom.ker f') (Ideal.comap f I)
    this : LE.le (RingHom.ker (Ideal.Quotient.mk (RingHom.ker f'))) p
    ⊢ Membership.mem (minimalPrimes (HasQuotient.Quotient R (RingHom.ker f'))) (Id …
  -/
  refine ⟨⟨?_, bot_le⟩, ?_⟩
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      I : Ideal S
      f : RingHom R S
      p : Ideal R
      H : Membership.mem (Ideal.comap f I).minimalPrimes p
      this✝ : p.IsPrime
      f' : RingHom R (HasQuotient.Quotient S I) := (Ideal.Quotient.mk I).comp f
      e : Eq (RingHom.ker f') (Ideal.comap f I)
      this : LE.le (RingHom.ker (Ideal.Quotient.mk (RingHom.ker f'))) p
      ⊢ (Ideal.map (Ideal.Quotient.mk (RingHom.ker f')) p).IsPrime
    -/
  · apply Ideal.map_isPrime_of_surjective _ this
    /-
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      I : Ideal S
      f : RingHom R S
      p : Ideal R
      H : Membership.mem (Ideal.comap f I).minimalPrimes p
      this✝ : p.IsPrime
      f' : RingHom R (HasQuotient.Quotient S I) := (Ideal.Quotient.mk I).comp f
      e : Eq (RingHom.ker f') (Ideal.comap f I)
      this : LE.le (RingHom.ker (Ideal.Quotient.mk (RingHom.ker f'))) p
      ⊢ Function.Surjective ⇑(Ideal.Quotient.mk (RingHom.ker f'))
    -/
    exact Ideal.Quotient.mk_surjective
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      I : Ideal S
      f : RingHom R S
      p : Ideal R
      H : Membership.mem (Ideal.comap f I).minimalPrimes p
      this✝ : p.IsPrime
      f' : RingHom R (HasQuotient.Quotient S I) := (Ideal.Quotient.mk I).comp f
      e : Eq (RingHom.ker f') (Ideal.comap f I)
      this : LE.le (RingHom.ker (Ideal.Quotient.mk (RingHom.ker f'))) p
      ⊢ ∀ ⦃y : Ideal (HasQuotient.Quotient R (RingHom.ker f'))⦄, (fun q => And q.IsP …
    -/
  · rintro q ⟨hq, -⟩ hq'
    rw [← Ideal.map_comap_of_surjective
        (Ideal.Quotient.mk (RingHom.ker ((Ideal.Quotient.mk I).comp f)))
        Ideal.Quotient.mk_surjective q]
    /-
      case refine_2.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      I : Ideal S
      f : RingHom R S
      p : Ideal R
      H : Membership.mem (Ideal.comap f I).minimalPrimes p
      this✝ : p.IsPrime
      f' : RingHom R (HasQuotient.Quotient S I) := (Ideal.Quotient.mk I).comp f
      e : Eq (RingHom.ker f') (Ideal.comap f I)
      this : LE.le (RingHom.ker (Ideal.Quotient.mk (RingHom.ker f'))) p
      q : Ideal (HasQuotient.Quotient R (RingHom.ker f'))
      hq : q.IsPrime
      hq' : LE.le q (Ideal.map (Ideal.Quotient.mk (RingHom.ker f')) p)
      ⊢ LE.le (Ideal.map (Ideal.Quotient.mk (RingHom.ker f')) p) (Ideal.map (Ideal.Q …
    -/
    apply Ideal.map_mono
    /-
      case refine_2.intro.h
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      I : Ideal S
      f : RingHom R S
      p : Ideal R
      H : Membership.mem (Ideal.comap f I).minimalPrimes p
      this✝ : p.IsPrime
      f' : RingHom R (HasQuotient.Quotient S I) := (Ideal.Quotient.mk I).comp f
      e : Eq (RingHom.ker f') (Ideal.comap f I)
      this : LE.le (RingHom.ker (Ideal.Quotient.mk (RingHom.ker f'))) p
      q : Ideal (HasQuotient.Quotient R (RingHom.ker f'))
      hq : q.IsPrime
      hq' : LE.le q (Ideal.map (Ideal.Quotient.mk (RingHom.ker f')) p)
      ⊢ LE.le p (Ideal.comap (Ideal.Quotient.mk (RingHom.ker ((Ideal.Quotient.mk I). …
    -/
    apply H.2
      /-
        case refine_2.intro.h.a
        R : Type u_1
        S : Type u_2
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        I : Ideal S
        f : RingHom R S
        p : Ideal R
        H : Membership.mem (Ideal.comap f I).minimalPrimes p
        this✝ : p.IsPrime
        f' : RingHom R (HasQuotient.Quotient S I) := (Ideal.Quotient.mk I).comp f
        e : Eq (RingHom.ker f') (Ideal.comap f I)
        this : LE.le (RingHom.ker (Ideal.Quotient.mk (RingHom.ker f'))) p
        q : Ideal (HasQuotient.Quotient R (RingHom.ker f'))
        hq : q.IsPrime
        hq' : LE.le q (Ideal.map (Ideal.Quotient.mk (RingHom.ker f')) p)
        ⊢ And (Ideal.comap (Ideal.Quotient.mk (RingHom.ker ((Ideal.Quotient.mk I).comp …
      -/
    · refine ⟨inferInstance, (Ideal.mk_ker.trans e).symm.trans_le (Ideal.comap_mono bot_le)⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.h.a
        R : Type u_1
        S : Type u_2
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        I : Ideal S
        f : RingHom R S
        p : Ideal R
        H : Membership.mem (Ideal.comap f I).minimalPrimes p
        this✝ : p.IsPrime
        f' : RingHom R (HasQuotient.Quotient S I) := (Ideal.Quotient.mk I).comp f
        e : Eq (RingHom.ker f') (Ideal.comap f I)
        this : LE.le (RingHom.ker (Ideal.Quotient.mk (RingHom.ker f'))) p
        q : Ideal (HasQuotient.Quotient R (RingHom.ker f'))
        hq : q.IsPrime
        hq' : LE.le q (Ideal.map (Ideal.Quotient.mk (RingHom.ker f')) p)
        ⊢ LE.le (Ideal.comap (Ideal.Quotient.mk (RingHom.ker ((Ideal.Quotient.mk I).co …
      -/
    · refine (Ideal.comap_mono hq').trans ?_
      /-
        case refine_2.intro.h.a
        R : Type u_1
        S : Type u_2
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        I : Ideal S
        f : RingHom R S
        p : Ideal R
        H : Membership.mem (Ideal.comap f I).minimalPrimes p
        this✝ : p.IsPrime
        f' : RingHom R (HasQuotient.Quotient S I) := (Ideal.Quotient.mk I).comp f
        e : Eq (RingHom.ker f') (Ideal.comap f I)
        this : LE.le (RingHom.ker (Ideal.Quotient.mk (RingHom.ker f'))) p
        q : Ideal (HasQuotient.Quotient R (RingHom.ker f'))
        hq : q.IsPrime
        hq' : LE.le q (Ideal.map (Ideal.Quotient.mk (RingHom.ker f')) p)
        ⊢ LE.le (Ideal.comap (Ideal.Quotient.mk (RingHom.ker ((Ideal.Quotient.mk I).co …
      -/
      rw [Ideal.comap_map_of_surjective]
      /-
        case refine_2.intro.h.a
        R : Type u_1
        S : Type u_2
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        I : Ideal S
        f : RingHom R S
        p : Ideal R
        H : Membership.mem (Ideal.comap f I).minimalPrimes p
        this✝ : p.IsPrime
        f' : RingHom R (HasQuotient.Quotient S I) := (Ideal.Quotient.mk I).comp f
        e : Eq (RingHom.ker f') (Ideal.comap f I)
        this : LE.le (RingHom.ker (Ideal.Quotient.mk (RingHom.ker f'))) p
        q : Ideal (HasQuotient.Quotient R (RingHom.ker f'))
        hq : q.IsPrime
        hq' : LE.le q (Ideal.map (Ideal.Quotient.mk (RingHom.ker f')) p)
        ⊢ LE.le (Max.max p (Ideal.comap (Ideal.Quotient.mk (RingHom.ker ((Ideal.Quotie …
      -/
      exacts [sup_le rfl.le this, Ideal.Quotient.mk_surjective]
      /-
        🎉 no goals
      -/


theorem Ideal.exists_minimalPrimes_comap_eq {I : Ideal S} (f : R →+* S) (p)
    (H : p ∈ (I.comap f).minimalPrimes) : ∃ p' ∈ I.minimalPrimes, Ideal.comap f p' = p := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    I : Ideal S
    f : RingHom R S
    p : Ideal R
    H : Membership.mem (Ideal.comap f I).minimalPrimes p
    ⊢ Exists fun p' => And (Membership.mem I.minimalPrimes p') (Eq (Ideal.comap f  …
  -/
  obtain ⟨p', h₁, h₂, h₃⟩ := Ideal.exists_comap_eq_of_mem_minimalPrimes f p H
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    I : Ideal S
    f : RingHom R S
    p : Ideal R
    H : Membership.mem (Ideal.comap f I).minimalPrimes p
    p' : Ideal S
    h₁ : p'.IsPrime
    h₂ : LE.le I p'
    h₃ : Eq (Ideal.comap f p') p
    ⊢ Exists fun p' => And (Membership.mem I.minimalPrimes p') (Eq (Ideal.comap f  …
  -/
  obtain ⟨q, hq, hq'⟩ := Ideal.exists_minimalPrimes_le h₂
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    I : Ideal S
    f : RingHom R S
    p : Ideal R
    H : Membership.mem (Ideal.comap f I).minimalPrimes p
    p' : Ideal S
    h₁ : p'.IsPrime
    h₂ : LE.le I p'
    h₃ : Eq (Ideal.comap f p') p
    q : Ideal S
    hq : Membership.mem I.minimalPrimes q
    hq' : LE.le q p'
    ⊢ Exists fun p' => And (Membership.mem I.minimalPrimes p') (Eq (Ideal.comap f  …
  -/
  refine ⟨q, hq, Eq.symm ?_⟩
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    I : Ideal S
    f : RingHom R S
    p : Ideal R
    H : Membership.mem (Ideal.comap f I).minimalPrimes p
    p' : Ideal S
    h₁ : p'.IsPrime
    h₂ : LE.le I p'
    h₃ : Eq (Ideal.comap f p') p
    q : Ideal S
    hq : Membership.mem I.minimalPrimes q
    hq' : LE.le q p'
    ⊢ Eq p (Ideal.comap f q)
  -/
  have := hq.1.1
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    I : Ideal S
    f : RingHom R S
    p : Ideal R
    H : Membership.mem (Ideal.comap f I).minimalPrimes p
    p' : Ideal S
    h₁ : p'.IsPrime
    h₂ : LE.le I p'
    h₃ : Eq (Ideal.comap f p') p
    q : Ideal S
    hq : Membership.mem I.minimalPrimes q
    hq' : LE.le q p'
    this : q.IsPrime
    ⊢ Eq p (Ideal.comap f q)
  -/
  have := (Ideal.comap_mono hq').trans_eq h₃
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    I : Ideal S
    f : RingHom R S
    p : Ideal R
    H : Membership.mem (Ideal.comap f I).minimalPrimes p
    p' : Ideal S
    h₁ : p'.IsPrime
    h₂ : LE.le I p'
    h₃ : Eq (Ideal.comap f p') p
    q : Ideal S
    hq : Membership.mem I.minimalPrimes q
    hq' : LE.le q p'
    this✝ : q.IsPrime
    this : LE.le (Ideal.comap f q) p
    ⊢ Eq p (Ideal.comap f q)
  -/
  exact (H.2 ⟨inferInstance, Ideal.comap_mono hq.1.2⟩ this).antisymm this
  /-
    🎉 no goals
  -/


theorem Ideal.minimal_primes_comap_of_surjective {f : R →+* S} (hf : Function.Surjective f)
    {I J : Ideal S} (h : J ∈ I.minimalPrimes) : J.comap f ∈ (I.comap f).minimalPrimes := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I J : Ideal S
    h : Membership.mem I.minimalPrimes J
    ⊢ Membership.mem (Ideal.comap f I).minimalPrimes (Ideal.comap f J)
  -/
  have := h.1.1
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I J : Ideal S
    h : Membership.mem I.minimalPrimes J
    this : J.IsPrime
    ⊢ Membership.mem (Ideal.comap f I).minimalPrimes (Ideal.comap f J)
  -/
  refine ⟨⟨inferInstance, Ideal.comap_mono h.1.2⟩, ?_⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I J : Ideal S
    h : Membership.mem I.minimalPrimes J
    this : J.IsPrime
    ⊢ ∀ ⦃y : Ideal R⦄, (fun q => And q.IsPrime (LE.le (Ideal.comap f I) q)) y → LE …
  -/
  rintro K ⟨hK, e₁⟩ e₂
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I J : Ideal S
    h : Membership.mem I.minimalPrimes J
    this : J.IsPrime
    K : Ideal R
    hK : K.IsPrime
    e₁ : LE.le (Ideal.comap f I) K
    e₂ : LE.le K (Ideal.comap f J)
    ⊢ LE.le (Ideal.comap f J) K
  -/
  have : RingHom.ker f ≤ K := (Ideal.comap_mono bot_le).trans e₁
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I J : Ideal S
    h : Membership.mem I.minimalPrimes J
    this✝ : J.IsPrime
    K : Ideal R
    hK : K.IsPrime
    e₁ : LE.le (Ideal.comap f I) K
    e₂ : LE.le K (Ideal.comap f J)
    this : LE.le (RingHom.ker f) K
    ⊢ LE.le (Ideal.comap f J) K
  -/
  rw [← sup_eq_left.mpr this, RingHom.ker_eq_comap_bot, ← Ideal.comap_map_of_surjective f hf]
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I J : Ideal S
    h : Membership.mem I.minimalPrimes J
    this✝ : J.IsPrime
    K : Ideal R
    hK : K.IsPrime
    e₁ : LE.le (Ideal.comap f I) K
    e₂ : LE.le K (Ideal.comap f J)
    this : LE.le (RingHom.ker f) K
    ⊢ LE.le (Ideal.comap f J) (Ideal.comap f (Ideal.map f K))
  -/
  apply Ideal.comap_mono _
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I J : Ideal S
    h : Membership.mem I.minimalPrimes J
    this✝ : J.IsPrime
    K : Ideal R
    hK : K.IsPrime
    e₁ : LE.le (Ideal.comap f I) K
    e₂ : LE.le K (Ideal.comap f J)
    this : LE.le (RingHom.ker f) K
    ⊢ LE.le J (Ideal.map f K)
  -/
  apply h.2 _ _
    /-
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I J : Ideal S
      h : Membership.mem I.minimalPrimes J
      this✝ : J.IsPrime
      K : Ideal R
      hK : K.IsPrime
      e₁ : LE.le (Ideal.comap f I) K
      e₂ : LE.le K (Ideal.comap f J)
      this : LE.le (RingHom.ker f) K
      ⊢ And (Ideal.map f K).IsPrime (LE.le I (Ideal.map f K))
    -/
  · exact ⟨Ideal.map_isPrime_of_surjective hf this, Ideal.le_map_of_comap_le_of_surjective f hf e₁⟩
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I J : Ideal S
      h : Membership.mem I.minimalPrimes J
      this✝ : J.IsPrime
      K : Ideal R
      hK : K.IsPrime
      e₁ : LE.le (Ideal.comap f I) K
      e₂ : LE.le K (Ideal.comap f J)
      this : LE.le (RingHom.ker f) K
      ⊢ LE.le (Ideal.map f K) J
    -/
  · exact Ideal.map_le_of_le_comap e₂
    /-
      🎉 no goals
    -/


theorem Ideal.comap_minimalPrimes_eq_of_surjective {f : R →+* S} (hf : Function.Surjective f)
    (I : Ideal S) : (I.comap f).minimalPrimes = Ideal.comap f '' I.minimalPrimes := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I : Ideal S
    ⊢ Eq (Ideal.comap f I).minimalPrimes (Set.image (Ideal.comap f) I.minimalPrimes)
  -/
  ext J
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I : Ideal S
    J : Ideal R
    ⊢ Iff (Membership.mem (Ideal.comap f I).minimalPrimes J) (Membership.mem (Set. …
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal S
      J : Ideal R
      ⊢ Membership.mem (Ideal.comap f I).minimalPrimes J → Membership.mem (Set.image …
    -/
  · intro H
    /-
      case h.mp
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal S
      J : Ideal R
      H : Membership.mem (Ideal.comap f I).minimalPrimes J
      ⊢ Membership.mem (Set.image (Ideal.comap f) I.minimalPrimes) J
    -/
    obtain ⟨p, h, rfl⟩ := Ideal.exists_minimalPrimes_comap_eq f J H
    /-
      case h.mp.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I p : Ideal S
      h : Membership.mem I.minimalPrimes p
      H : Membership.mem (Ideal.comap f I).minimalPrimes (Ideal.comap f p)
      ⊢ Membership.mem (Set.image (Ideal.comap f) I.minimalPrimes) (Ideal.comap f p)
    -/
    exact ⟨p, h, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal S
      J : Ideal R
      ⊢ Membership.mem (Set.image (Ideal.comap f) I.minimalPrimes) J → Membership.me …
    -/
  · rintro ⟨J, hJ, rfl⟩
    /-
      case h.mpr.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I J : Ideal S
      hJ : Membership.mem I.minimalPrimes J
      ⊢ Membership.mem (Ideal.comap f I).minimalPrimes (Ideal.comap f J)
    -/
    exact Ideal.minimal_primes_comap_of_surjective hf hJ
    /-
      🎉 no goals
    -/


theorem Ideal.minimalPrimes_eq_comap :
    I.minimalPrimes = Ideal.comap (Ideal.Quotient.mk I) '' minimalPrimes (R ⧸ I) := by
  rw [minimalPrimes, ← Ideal.comap_minimalPrimes_eq_of_surjective Ideal.Quotient.mk_surjective,
    ← RingHom.ker_eq_comap_bot, Ideal.mk_ker]


theorem Ideal.minimalPrimes_eq_subsingleton (hI : I.IsPrimary) : I.minimalPrimes = {I.radical} := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    hI : I.IsPrimary
    ⊢ Eq I.minimalPrimes (Singleton.singleton I.radical)
  -/
  ext J
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    hI : I.IsPrimary
    J : Ideal R
    ⊢ Iff (Membership.mem I.minimalPrimes J) (Membership.mem (Singleton.singleton  …
  -/
  constructor
  · exact fun H =>
      let e := H.1.1.radical_le_iff.mpr H.1.2
      (H.2 ⟨Ideal.isPrime_radical hI, Ideal.le_radical⟩ e).antisymm e
    /-
      case h.mpr
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      hI : I.IsPrimary
      J : Ideal R
      ⊢ Membership.mem (Singleton.singleton I.radical) J → Membership.mem I.minimalP …
    -/
  · rintro (rfl : J = I.radical)
    /-
      case h.mpr
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      hI : I.IsPrimary
      ⊢ Membership.mem I.minimalPrimes I.radical
    -/
    exact ⟨⟨Ideal.isPrime_radical hI, Ideal.le_radical⟩, fun _ H _ => H.1.radical_le_iff.mpr H.2⟩
    /-
      🎉 no goals
    -/


theorem Ideal.minimalPrimes_eq_subsingleton_self [I.IsPrime] : I.minimalPrimes = {I} := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : I.IsPrime
    ⊢ Eq I.minimalPrimes (Singleton.singleton I)
  -/
  ext J
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : I.IsPrime
    J : Ideal R
    ⊢ Iff (Membership.mem I.minimalPrimes J) (Membership.mem (Singleton.singleton  …
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      inst✝¹ : CommRing R
      I : Ideal R
      inst✝ : I.IsPrime
      J : Ideal R
      ⊢ Membership.mem I.minimalPrimes J → Membership.mem (Singleton.singleton I) J
    -/
  · exact fun H => (H.2 ⟨inferInstance, rfl.le⟩ H.1.2).antisymm H.1.2
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u_1
      inst✝¹ : CommRing R
      I : Ideal R
      inst✝ : I.IsPrime
      J : Ideal R
      ⊢ Membership.mem (Singleton.singleton I) J → Membership.mem I.minimalPrimes J
    -/
  · rintro (rfl : J = I)
    /-
      case h.mpr
      R : Type u_1
      inst✝¹ : CommRing R
      J : Ideal R
      inst✝ : J.IsPrime
      ⊢ Membership.mem J.minimalPrimes J
    -/
    exact ⟨⟨inferInstance, rfl.le⟩, fun _ h _ => h.2⟩
    /-
      🎉 no goals
    -/


theorem _root_.IsLocalization.AtPrime.prime_unique_of_minimal {S} [CommSemiring S] [Algebra R S]
    [IsLocalization.AtPrime S I] {J K : Ideal S} [J.IsPrime] [K.IsPrime] : J = K :=
  haveI : Subsingleton {i : Ideal R // i.IsPrime ∧ i ≤ I} := ⟨fun i₁ i₂ ↦ Subtype.ext <| by
    /-
      R : Type u_1
      inst✝⁵ : CommSemiring R
      I : Ideal R
      hI : I.IsPrime
      hMin : Membership.mem (minimalPrimes R) I
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      inst✝² : IsLocalization.AtPrime S I
      J K : Ideal S
      inst✝¹ : J.IsPrime
      inst✝ : K.IsPrime
      i₁ i₂ : Subtype fun i => And i.IsPrime (LE.le i I)
      ⊢ Eq ↑i₁ ↑i₂
    -/
    rw [minimalPrimes_eq_minimals, Set.mem_setOf] at hMin
    /-
      R : Type u_1
      inst✝⁵ : CommSemiring R
      I : Ideal R
      hI : I.IsPrime
      hMin : Minimal Ideal.IsPrime I
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      inst✝² : IsLocalization.AtPrime S I
      J K : Ideal S
      inst✝¹ : J.IsPrime
      inst✝ : K.IsPrime
      i₁ i₂ : Subtype fun i => And i.IsPrime (LE.le i I)
      ⊢ Eq ↑i₁ ↑i₂
    -/
    rw [hMin.eq_of_le i₁.2.1 i₁.2.2, hMin.eq_of_le i₂.2.1 i₂.2.2]⟩
    /-
      🎉 no goals
    -/
  Subtype.ext_iff.mp <| (IsLocalization.AtPrime.orderIsoOfPrime S I).injective
    (a₁ := ⟨J, ‹_›⟩) (a₂ := ⟨K, ‹_›⟩) (Subsingleton.elim _ _)


theorem prime_unique_of_minimal (J : Ideal (Localization I.primeCompl)) [J.IsPrime] :
    J = IsLocalRing.maximalIdeal (Localization I.primeCompl) :=
  IsLocalization.AtPrime.prime_unique_of_minimal hMin


theorem nilpotent_iff_mem_maximal_of_minimal {x : _} :
    IsNilpotent x ↔ x ∈ IsLocalRing.maximalIdeal (Localization I.primeCompl) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    I : Ideal R
    hI : I.IsPrime
    hMin : Membership.mem (minimalPrimes R) I
    x : Localization I.primeCompl
    ⊢ Iff (IsNilpotent x) (Membership.mem (IsLocalRing.maximalIdeal (Localization  …
  -/
  rw [nilpotent_iff_mem_prime]
  exact ⟨(· (IsLocalRing.maximalIdeal _) (Ideal.IsMaximal.isPrime' _)), fun _ J _ =>
    by simpa [prime_unique_of_minimal hMin J]⟩


theorem nilpotent_iff_not_unit_of_minimal {x : Localization I.primeCompl} :
    IsNilpotent x ↔ x ∈ nonunits _ := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    I : Ideal R
    hI : I.IsPrime
    hMin : Membership.mem (minimalPrimes R) I
    x : Localization I.primeCompl
    ⊢ Iff (IsNilpotent x) (Membership.mem (nonunits (Localization I.primeCompl)) x)
  -/
  simpa only [← IsLocalRing.mem_maximalIdeal] using nilpotent_iff_mem_maximal_of_minimal hMin
  /-
    🎉 no goals
  -/


