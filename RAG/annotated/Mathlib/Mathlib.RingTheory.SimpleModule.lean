/-- A module is simple when it has only two submodules, `⊥` and `⊤`. -/
abbrev IsSimpleModule :=
  IsSimpleOrder (Submodule R M)


/-- A module is semisimple when every submodule has a complement, or equivalently, the module
  is a direct sum of simple modules. -/
abbrev IsSemisimpleModule :=
  ComplementedLattice (Submodule R M)


/-- A ring is semisimple if it is semisimple as a module over itself. -/
abbrev IsSemisimpleRing := IsSemisimpleModule R R


theorem RingEquiv.isSemisimpleRing (e : R ≃+* S) [IsSemisimpleRing R] : IsSemisimpleRing S :=
  (Submodule.orderIsoMapComap e.toSemilinearEquiv).complementedLattice

-- Making this an instance causes the linter to complain of "dangerous instances"

theorem IsSimpleModule.nontrivial [IsSimpleModule R M] : Nontrivial M :=
  ⟨⟨0, by
      /-
        R : Type u_2
        inst✝³ : Ring R
        M : Type u_4
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsSimpleModule R M
        ⊢ Exists fun y => Ne 0 y
      -/
      have h : (⊥ : Submodule R M) ≠ ⊤ := bot_ne_top
      /-
        R : Type u_2
        inst✝³ : Ring R
        M : Type u_4
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsSimpleModule R M
        h : Ne Bot.bot Top.top
        ⊢ Exists fun y => Ne 0 y
      -/
      contrapose! h
      /-
        R : Type u_2
        inst✝³ : Ring R
        M : Type u_4
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsSimpleModule R M
        h : ∀ (y : M), Eq 0 y
        ⊢ Eq Bot.bot Top.top
      -/
      ext x
      /-
        case h
        R : Type u_2
        inst✝³ : Ring R
        M : Type u_4
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsSimpleModule R M
        h : ∀ (y : M), Eq 0 y
        x : M
        ⊢ Iff (Membership.mem Bot.bot x) (Membership.mem Top.top x)
      -/
      simp [Submodule.mem_bot, Submodule.mem_top, h x]⟩⟩
      /-
        🎉 no goals
      -/


theorem LinearMap.isSimpleModule_iff_of_bijective [Module S N] {σ : R →+* S} [RingHomSurjective σ]
    (l : M →ₛₗ[σ] N) (hl : Function.Bijective l) : IsSimpleModule R M ↔ IsSimpleModule S N :=
  (Submodule.orderIsoMapComapOfBijective l hl).isSimpleOrder_iff


theorem IsSimpleModule.congr (l : M ≃ₗ[R] N) [IsSimpleModule R N] : IsSimpleModule R M :=
  (Submodule.orderIsoMapComap l).isSimpleOrder


theorem isSimpleModule_iff_isAtom : IsSimpleModule R m ↔ IsAtom m := by
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : Submodule R M
    ⊢ Iff (IsSimpleModule R (Subtype fun x => Membership.mem m x)) (IsAtom m)
  -/
  rw [← Set.isSimpleOrder_Iic_iff_isAtom]
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : Submodule R M
    ⊢ Iff (IsSimpleModule R (Subtype fun x => Membership.mem m x)) (IsSimpleOrder  …
  -/
  exact m.mapIic.isSimpleOrder_iff
  /-
    🎉 no goals
  -/


theorem isSimpleModule_iff_isCoatom : IsSimpleModule R (M ⧸ m) ↔ IsCoatom m := by
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : Submodule R M
    ⊢ Iff (IsSimpleModule R (HasQuotient.Quotient M m)) (IsCoatom m)
  -/
  rw [← Set.isSimpleOrder_Ici_iff_isCoatom]
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : Submodule R M
    ⊢ Iff (IsSimpleModule R (HasQuotient.Quotient M m)) (IsSimpleOrder ↑(Set.Ici m))
  -/
  apply OrderIso.isSimpleOrder_iff
  /-
    case f
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : Submodule R M
    ⊢ OrderIso (Submodule R (HasQuotient.Quotient M m)) ↑(Set.Ici m)
  -/
  exact Submodule.comapMkQRelIso m
  /-
    🎉 no goals
  -/


theorem covBy_iff_quot_is_simple {A B : Submodule R M} (hAB : A ≤ B) :
    A ⋖ B ↔ IsSimpleModule R (B ⧸ Submodule.comap B.subtype A) := by
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A B : Submodule R M
    hAB : LE.le A B
    ⊢ Iff (CovBy A B) (IsSimpleModule R (HasQuotient.Quotient (Subtype fun x => Me …
  -/
  set f : Submodule R B ≃o Set.Iic B := B.mapIic with hf
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A B : Submodule R M
    hAB : LE.le A B
    f : OrderIso (Submodule R (Subtype fun x => Membership.mem B x)) ↑(Set.Iic B)  …
    hf : Eq f B.mapIic
    ⊢ Iff (CovBy A B) (IsSimpleModule R (HasQuotient.Quotient (Subtype fun x => Me …
  -/
  rw [covBy_iff_coatom_Iic hAB, isSimpleModule_iff_isCoatom, ← OrderIso.isCoatom_iff f, hf]
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A B : Submodule R M
    hAB : LE.le A B
    f : OrderIso (Submodule R (Subtype fun x => Membership.mem B x)) ↑(Set.Iic B)  …
    hf : Eq f B.mapIic
    ⊢ Iff (IsCoatom ⟨A, hAB⟩) (IsCoatom (B.mapIic (Submodule.comap B.subtype A)))
  -/
  simp [-OrderIso.isCoatom_iff, Submodule.map_comap_subtype, inf_eq_right.2 hAB]
  /-
    🎉 no goals
  -/


@[simp]
theorem isAtom [IsSimpleModule R m] : IsAtom m :=
  isSimpleModule_iff_isAtom.1 ‹_›


theorem span_singleton_eq_top {m : M} (hm : m ≠ 0) : Submodule.span R {m} = ⊤ :=
  (eq_bot_or_eq_top _).resolve_left fun h ↦ hm (h.le <| Submodule.mem_span_singleton_self m)


instance (S : Submodule R M) : S.IsPrincipal where
  principal' := by
    /-
      ι : Type u_1
      R : Type u_2
      S✝ : Type u_3
      inst✝⁶ : Ring R
      inst✝⁵ : Ring S✝
      M : Type u_4
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      m : Submodule R M
      N : Type u_5
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      inst✝ : IsSimpleModule R M
      S : Submodule R M
      ⊢ Exists fun a => Eq S (Submodule.span R (Singleton.singleton a))
    -/
    obtain rfl | rfl := eq_bot_or_eq_top S
      /-
        case inl
        ι : Type u_1
        R : Type u_2
        S : Type u_3
        inst✝⁶ : Ring R
        inst✝⁵ : Ring S
        M : Type u_4
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        m : Submodule R M
        N : Type u_5
        inst✝² : AddCommGroup N
        inst✝¹ : Module R N
        inst✝ : IsSimpleModule R M
        ⊢ Exists fun a => Eq Bot.bot (Submodule.span R (Singleton.singleton a))
      -/
    · exact ⟨0, Submodule.span_zero.symm⟩
      /-
        🎉 no goals
      -/
    /-
      case inr
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝⁶ : Ring R
      inst✝⁵ : Ring S
      M : Type u_4
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      m : Submodule R M
      N : Type u_5
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      inst✝ : IsSimpleModule R M
      ⊢ Exists fun a => Eq Top.top (Submodule.span R (Singleton.singleton a))
    -/
    have := IsSimpleModule.nontrivial R M
    /-
      case inr
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝⁶ : Ring R
      inst✝⁵ : Ring S
      M : Type u_4
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      m : Submodule R M
      N : Type u_5
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      inst✝ : IsSimpleModule R M
      this : Nontrivial M
      ⊢ Exists fun a => Eq Top.top (Submodule.span R (Singleton.singleton a))
    -/
    have ⟨m, hm⟩ := exists_ne (0 : M)
    /-
      case inr
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝⁶ : Ring R
      inst✝⁵ : Ring S
      M : Type u_4
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      m✝ : Submodule R M
      N : Type u_5
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      inst✝ : IsSimpleModule R M
      this : Nontrivial M
      m : M
      hm : Ne m 0
      ⊢ Exists fun a => Eq Top.top (Submodule.span R (Singleton.singleton a))
    -/
    exact ⟨m, (span_singleton_eq_top R hm).symm⟩
    /-
      🎉 no goals
    -/


theorem toSpanSingleton_surjective {m : M} (hm : m ≠ 0) :
    Function.Surjective (toSpanSingleton R M m) := by
  /-
    R : Type u_2
    inst✝³ : Ring R
    M : Type u_4
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSimpleModule R M
    m : M
    hm : Ne m 0
    ⊢ Function.Surjective ⇑(LinearMap.toSpanSingleton R M m)
  -/
  rw [← range_eq_top, ← span_singleton_eq_range, span_singleton_eq_top R hm]
  /-
    🎉 no goals
  -/


theorem ker_toSpanSingleton_isMaximal {m : M} (hm : m ≠ 0) :
    Ideal.IsMaximal (ker (toSpanSingleton R M m)) := by
  /-
    R : Type u_2
    inst✝³ : Ring R
    M : Type u_4
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSimpleModule R M
    m : M
    hm : Ne m 0
    ⊢ Ideal.IsMaximal (LinearMap.ker (LinearMap.toSpanSingleton R M m))
  -/
  rw [Ideal.isMaximal_def, ← isSimpleModule_iff_isCoatom]
  /-
    R : Type u_2
    inst✝³ : Ring R
    M : Type u_4
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSimpleModule R M
    m : M
    hm : Ne m 0
    ⊢ IsSimpleModule R (HasQuotient.Quotient R (LinearMap.ker (LinearMap.toSpanSin …
  -/
  exact congr (quotKerEquivOfSurjective _ <| toSpanSingleton_surjective R hm)
  /-
    🎉 no goals
  -/


instance : IsNoetherian R M := isNoetherian_iff'.mpr inferInstance


open IsSimpleModule in
/-- A module is simple iff it's isomorphic to the quotient of the ring by a maximal left ideal
(not necessarily unique if the ring is not commutative). -/
theorem isSimpleModule_iff_quot_maximal :
    IsSimpleModule R M ↔ ∃ I : Ideal R, I.IsMaximal ∧ Nonempty (M ≃ₗ[R] R ⧸ I) := by
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (IsSimpleModule R M) (Exists fun I => And I.IsMaximal (Nonempty (LinearE …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨I, ⟨coatom⟩, ⟨equiv⟩⟩ ↦ ?_⟩
    /-
      case refine_1
      R : Type u_2
      inst✝² : Ring R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      h : IsSimpleModule R M
      ⊢ Exists fun I => And I.IsMaximal (Nonempty (LinearEquiv (RingHom.id R) M (Has …
    -/
  · have := IsSimpleModule.nontrivial R M
    /-
      case refine_1
      R : Type u_2
      inst✝² : Ring R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      h : IsSimpleModule R M
      this : Nontrivial M
      ⊢ Exists fun I => And I.IsMaximal (Nonempty (LinearEquiv (RingHom.id R) M (Has …
    -/
    have ⟨m, hm⟩ := exists_ne (0 : M)
    exact ⟨_, ker_toSpanSingleton_isMaximal R hm,
      ⟨(LinearMap.quotKerEquivOfSurjective _ <| toSpanSingleton_surjective R hm).symm⟩⟩
    /-
      case refine_2
      R : Type u_2
      inst✝² : Ring R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x✝ : Exists fun I => And I.IsMaximal (Nonempty (LinearEquiv (RingHom.id R) M ( …
      I : Ideal R
      coatom : IsCoatom I
      equiv : LinearEquiv (RingHom.id R) M (HasQuotient.Quotient R I)
      ⊢ IsSimpleModule R M
    -/
  · convert congr equiv; rwa [isSimpleModule_iff_isCoatom]
                         /-
                           🎉 no goals
                         -/


/-- In general, the annihilator of a simple module is called a primitive ideal, and it is
always a two-sided prime ideal, but mathlib's `Ideal.IsPrime` is not the correct definition
for noncommutative rings. -/
theorem IsSimpleModule.annihilator_isMaximal {R} [CommRing R] [Module R M]
    [simple : IsSimpleModule R M] : (Module.annihilator R M).IsMaximal := by
  /-
    M : Type u_4
    inst✝² : AddCommGroup M
    R : Type u_6
    inst✝¹ : CommRing R
    inst✝ : Module R M
    simple : IsSimpleModule R M
    ⊢ (Module.annihilator R M).IsMaximal
  -/
  have ⟨I, max, ⟨e⟩⟩ := isSimpleModule_iff_quot_maximal.mp simple
  /-
    M : Type u_4
    inst✝² : AddCommGroup M
    R : Type u_6
    inst✝¹ : CommRing R
    inst✝ : Module R M
    simple : IsSimpleModule R M
    I : Ideal R
    max : I.IsMaximal
    e : LinearEquiv (RingHom.id R) M (HasQuotient.Quotient R I)
    ⊢ (Module.annihilator R M).IsMaximal
  -/
  rwa [e.annihilator_eq, I.annihilator_quotient]
  /-
    🎉 no goals
  -/


theorem isSimpleModule_iff_toSpanSingleton_surjective : IsSimpleModule R M ↔
    Nontrivial M ∧ ∀ x : M, x ≠ 0 → Function.Surjective (LinearMap.toSpanSingleton R M x) :=
  ⟨fun h ↦ ⟨h.nontrivial, fun _ ↦ h.toSpanSingleton_surjective⟩, fun ⟨_, h⟩ ↦
    ⟨fun m ↦ or_iff_not_imp_left.mpr fun ne_bot ↦
      have ⟨x, hxm, hx0⟩ := m.ne_bot_iff.mp ne_bot
                                 /-
                                   R : Type u_2
                                   inst✝² : Ring R
                                   M : Type u_4
                                   inst✝¹ : AddCommGroup M
                                   inst✝ : Module R M
                                   x✝¹ : And (Nontrivial M) (∀ (x : M), Ne x 0 → Function.Surjective ⇑(LinearMap. …
                                   left✝ : Nontrivial M
                                   h : ∀ (x : M), Ne x 0 → Function.Surjective ⇑(LinearMap.toSpanSingleton R M x)
                                   m : Submodule R M
                                   ne_bot : Not (Eq m Bot.bot)
                                   x : M
                                   hxm : Membership.mem m x
                                   hx0 : Ne x 0
                                   z : M
                                   x✝ : Membership.mem Top.top z
                                   ⊢ Membership.mem m z
                                 -/
      top_unique <| fun z _ ↦ by obtain ⟨y, rfl⟩ := h x hx0 z; exact m.smul_mem _ hxm⟩⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- A ring is a simple module over itself iff it is a division ring. -/
theorem isSimpleModule_self_iff_isUnit :
    IsSimpleModule R R ↔ Nontrivial R ∧ ∀ x : R, x ≠ 0 → IsUnit x :=
  isSimpleModule_iff_toSpanSingleton_surjective.trans <| and_congr_right fun _ ↦ by
    /-
      R : Type u_2
      inst✝ : Ring R
      x✝ : Nontrivial R
      ⊢ Iff (∀ (x : R), Ne x 0 → Function.Surjective ⇑(LinearMap.toSpanSingleton R R …
    -/
    refine ⟨fun h x hx ↦ ?_, fun h x hx ↦ (h x hx).unit.mulRight_bijective.surjective⟩
    /-
      R : Type u_2
      inst✝ : Ring R
      x✝ : Nontrivial R
      h : ∀ (x : R), Ne x 0 → Function.Surjective ⇑(LinearMap.toSpanSingleton R R x)
      x : R
      hx : Ne x 0
      ⊢ IsUnit x
    -/
    obtain ⟨y, hyx : y * x = 1⟩ := h x hx 1
    /-
      case intro
      R : Type u_2
      inst✝ : Ring R
      x✝ : Nontrivial R
      h : ∀ (x : R), Ne x 0 → Function.Surjective ⇑(LinearMap.toSpanSingleton R R x)
      x : R
      hx : Ne x 0
      y : R
      hyx : Eq (HMul.hMul y x) 1
      ⊢ IsUnit x
    -/
    have hy : y ≠ 0 := left_ne_zero_of_mul (hyx.symm ▸ one_ne_zero)
    /-
      case intro
      R : Type u_2
      inst✝ : Ring R
      x✝ : Nontrivial R
      h : ∀ (x : R), Ne x 0 → Function.Surjective ⇑(LinearMap.toSpanSingleton R R x)
      x : R
      hx : Ne x 0
      y : R
      hyx : Eq (HMul.hMul y x) 1
      hy : Ne y 0
      ⊢ IsUnit x
    -/
    obtain ⟨z, hzy : z * y = 1⟩ := h y hy 1
    /-
      case intro.intro
      R : Type u_2
      inst✝ : Ring R
      x✝ : Nontrivial R
      h : ∀ (x : R), Ne x 0 → Function.Surjective ⇑(LinearMap.toSpanSingleton R R x)
      x : R
      hx : Ne x 0
      y : R
      hyx : Eq (HMul.hMul y x) 1
      hy : Ne y 0
      z : R
      hzy : Eq (HMul.hMul z y) 1
      ⊢ IsUnit x
    -/
    exact ⟨⟨x, y, left_inv_eq_right_inv hzy hyx ▸ hzy, hyx⟩, rfl⟩
    /-
      🎉 no goals
    -/


theorem isSimpleModule_iff_finrank_eq_one {R} [DivisionRing R] [Module R M] :
    IsSimpleModule R M ↔ Module.finrank R M = 1 :=
  ⟨fun h ↦ have := h.nontrivial; have ⟨v, hv⟩ := exists_ne (0 : M)
    (finrank_eq_one_iff_of_nonzero' v hv).mpr (IsSimpleModule.toSpanSingleton_surjective R hv),
  is_simple_module_of_finrank_eq_one⟩


theorem IsSemisimpleModule.of_sSup_simples_eq_top
    (h : sSup { m : Submodule R M | IsSimpleModule R m } = ⊤) : IsSemisimpleModule R M :=
                                               /-
                                                 R : Type u_2
                                                 inst✝² : Ring R
                                                 M : Type u_4
                                                 inst✝¹ : AddCommGroup M
                                                 inst✝ : Module R M
                                                 h : Eq (SupSet.sSup (setOf fun m => IsSimpleModule R (Subtype fun x => Members …
                                                 ⊢ Eq (SupSet.sSup (setOf fun a => IsAtom a)) Top.top
                                               -/
  complementedLattice_of_sSup_atoms_eq_top (by simp_rw [← h, isSimpleModule_iff_isAtom])
                                               /-
                                                 🎉 no goals
                                               -/


@[deprecated (since := "2024-03-05")]
alias is_semisimple_of_sSup_simples_eq_top := IsSemisimpleModule.of_sSup_simples_eq_top


theorem eq_bot_or_exists_simple_le (N : Submodule R M) : N = ⊥ ∨ ∃ m ≤ N, IsSimpleModule R m := by
  /-
    R : Type u_2
    inst✝³ : Ring R
    M : Type u_4
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    N : Submodule R M
    ⊢ Or (Eq N Bot.bot) (Exists fun m => And (LE.le m N) (IsSimpleModule R (Subtyp …
  -/
  simpa only [isSimpleModule_iff_isAtom, and_comm] using eq_bot_or_exists_atom_le _
  /-
    🎉 no goals
  -/


theorem sSup_simples_le (N : Submodule R M) :
    sSup { m : Submodule R M | IsSimpleModule R m ∧ m ≤ N } = N := by
  /-
    R : Type u_2
    inst✝³ : Ring R
    M : Type u_4
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    N : Submodule R M
    ⊢ Eq (SupSet.sSup (setOf fun m => And (IsSimpleModule R (Subtype fun x => Memb …
  -/
  simpa only [isSimpleModule_iff_isAtom] using sSup_atoms_le_eq _
  /-
    🎉 no goals
  -/


theorem exists_simple_submodule [Nontrivial M] : ∃ m : Submodule R M, IsSimpleModule R m := by
  /-
    R : Type u_2
    inst✝⁴ : Ring R
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsSemisimpleModule R M
    inst✝ : Nontrivial M
    ⊢ Exists fun m => IsSimpleModule R (Subtype fun x => Membership.mem m x)
  -/
  simpa only [isSimpleModule_iff_isAtom] using IsAtomic.exists_atom _
  /-
    🎉 no goals
  -/


theorem sSup_simples_eq_top : sSup { m : Submodule R M | IsSimpleModule R m } = ⊤ := by
  /-
    R : Type u_2
    inst✝³ : Ring R
    M : Type u_4
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    ⊢ Eq (SupSet.sSup (setOf fun m => IsSimpleModule R (Subtype fun x => Membershi …
  -/
  simpa only [isSimpleModule_iff_isAtom] using sSup_atoms_eq_top
  /-
    🎉 no goals
  -/


theorem exists_sSupIndep_sSup_simples_eq_top :
    ∃ s : Set (Submodule R M), sSupIndep s ∧ sSup s = ⊤ ∧ ∀ m ∈ s, IsSimpleModule R m := by
  /-
    R : Type u_2
    inst✝³ : Ring R
    M : Type u_4
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    ⊢ Exists fun s => And (sSupIndep s) (And (Eq (SupSet.sSup s) Top.top) (∀ (m :  …
  -/
  have := sSup_simples_eq_top R M
  /-
    R : Type u_2
    inst✝³ : Ring R
    M : Type u_4
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    this : Eq (SupSet.sSup (setOf fun m => IsSimpleModule R (Subtype fun x => Memb …
    ⊢ Exists fun s => And (sSupIndep s) (And (Eq (SupSet.sSup s) Top.top) (∀ (m :  …
  -/
  simp_rw [isSimpleModule_iff_isAtom] at this ⊢
  /-
    R : Type u_2
    inst✝³ : Ring R
    M : Type u_4
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    this : Eq (SupSet.sSup (setOf fun m => IsAtom m)) Top.top
    ⊢ Exists fun s => And (sSupIndep s) (And (Eq (SupSet.sSup s) Top.top) (∀ (m :  …
  -/
  exact exists_sSupIndep_of_sSup_atoms_eq_top this
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias exists_setIndependent_sSup_simples_eq_top := exists_sSupIndep_sSup_simples_eq_top


/-- The annihilator of a semisimple module over a commutative ring is a radical ideal. -/
theorem annihilator_isRadical (R) [CommRing R] [Module R M] [IsSemisimpleModule R M] :
    (Module.annihilator R M).IsRadical := by
  /-
    M : Type u_4
    inst✝³ : AddCommGroup M
    R : Type u_6
    inst✝² : CommRing R
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    ⊢ (Module.annihilator R M).IsRadical
  -/
  rw [← Submodule.annihilator_top, ← sSup_simples_eq_top, sSup_eq_iSup', Submodule.annihilator_iSup]
  /-
    M : Type u_4
    inst✝³ : AddCommGroup M
    R : Type u_6
    inst✝² : CommRing R
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    ⊢ (iInf fun i => (↑i).annihilator).IsRadical
  -/
  exact Ideal.isRadical_iInf _ fun i ↦ (i.2.annihilator_isMaximal).isPrime.isRadical
  /-
    🎉 no goals
  -/


instance submodule {m : Submodule R M} : IsSemisimpleModule R m :=
  m.mapIic.complementedLattice_iff.2 IsModularLattice.complementedLattice_Iic


theorem congr (e : N ≃ₗ[R] M) : IsSemisimpleModule R N :=
  (Submodule.orderIsoMapComap e.symm).complementedLattice


instance quotient : IsSemisimpleModule R (M ⧸ m) :=
  have ⟨P, compl⟩ := exists_isCompl m
  .congr (m.quotientEquivOfIsCompl P compl)


instance (priority := low) [Module.Finite R M] : IsNoetherian R M where
  noetherian m := have ⟨P, compl⟩ := exists_isCompl m
    Module.Finite.iff_fg.mp (Module.Finite.equiv <| P.quotientEquivOfIsCompl m compl.symm)

-- does not work as an instance, not sure why

protected theorem range (f : M →ₗ[R] N) : IsSemisimpleModule R (range f) :=
  .congr (quotKerEquivRange _).symm


theorem _root_.LinearMap.isSemisimpleModule_iff_of_bijective
    [RingHomSurjective σ] (hl : Function.Bijective l) :
    IsSemisimpleModule R M' ↔ IsSemisimpleModule S N' :=
  (Submodule.orderIsoMapComapOfBijective l hl).complementedLattice_iff

-- TODO: generalize Submodule.equivMapOfInjective from InvPair to RingHomSurjective

/-- A module is semisimple iff it is generated by its simple submodules. -/
theorem sSup_simples_eq_top_iff_isSemisimpleModule :
    sSup { m : Submodule R M | IsSimpleModule R m } = ⊤ ↔ IsSemisimpleModule R M :=
  ⟨.of_sSup_simples_eq_top, fun _ ↦ IsSemisimpleModule.sSup_simples_eq_top _ _⟩


@[deprecated (since := "2024-03-05")]
alias is_semisimple_iff_top_eq_sSup_simples := sSup_simples_eq_top_iff_isSemisimpleModule


/-- A module generated by semisimple submodules is itself semisimple. -/
lemma isSemisimpleModule_of_isSemisimpleModule_submodule {s : Set ι} {p : ι → Submodule R M}
    (hp : ∀ i ∈ s, IsSemisimpleModule R (p i)) (hp' : ⨆ i ∈ s, p i = ⊤) :
    IsSemisimpleModule R M := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set ι
    p : ι → Submodule R M
    hp : ∀ (i : ι), Membership.mem s i → IsSemisimpleModule R (Subtype fun x => Me …
    hp' : Eq (iSup fun i => iSup fun h => p i) Top.top
    ⊢ IsSemisimpleModule R M
  -/
  refine complementedLattice_of_complementedLattice_Iic (fun i hi ↦ ?_) hp'
  /-
    ι : Type u_1
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set ι
    p : ι → Submodule R M
    hp : ∀ (i : ι), Membership.mem s i → IsSemisimpleModule R (Subtype fun x => Me …
    hp' : Eq (iSup fun i => iSup fun h => p i) Top.top
    i : ι
    hi : Membership.mem s i
    ⊢ ComplementedLattice ↑(Set.Iic (p i))
  -/
  simpa only [← (p i).mapIic.complementedLattice_iff] using hp i hi
  /-
    🎉 no goals
  -/


lemma isSemisimpleModule_biSup_of_isSemisimpleModule_submodule {s : Set ι} {p : ι → Submodule R M}
    (hp : ∀ i ∈ s, IsSemisimpleModule R (p i)) :
    IsSemisimpleModule R ↥(⨆ i ∈ s, p i) := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set ι
    p : ι → Submodule R M
    hp : ∀ (i : ι), Membership.mem s i → IsSemisimpleModule R (Subtype fun x => Me …
    ⊢ IsSemisimpleModule R (Subtype fun x => Membership.mem (iSup fun i => iSup fu …
  -/
  let q := ⨆ i ∈ s, p i
  /-
    ι : Type u_1
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set ι
    p : ι → Submodule R M
    hp : ∀ (i : ι), Membership.mem s i → IsSemisimpleModule R (Subtype fun x => Me …
    q : Submodule R M := iSup fun i => iSup fun h => p i
    ⊢ IsSemisimpleModule R (Subtype fun x => Membership.mem (iSup fun i => iSup fu …
  -/
  let p' : ι → Submodule R q := fun i ↦ (p i).comap q.subtype
  have hp₀ : ∀ i ∈ s, p i ≤ LinearMap.range q.subtype := fun i hi ↦ by
    simpa only [Submodule.range_subtype] using le_biSup _ hi
  have hp₁ : ∀ i ∈ s, IsSemisimpleModule R (p' i) := fun i hi ↦ by
    let e : p' i ≃ₗ[R] p i := (p i).comap_equiv_self_of_inj_of_le q.injective_subtype (hp₀ i hi)
    exact (Submodule.orderIsoMapComap e).complementedLattice_iff.mpr <| hp i hi
  have hp₂ : ⨆ i ∈ s, p' i = ⊤ := by
    apply Submodule.map_injective_of_injective q.injective_subtype
    simp_rw [Submodule.map_top, Submodule.range_subtype, Submodule.map_iSup]
    exact biSup_congr fun i hi ↦ Submodule.map_comap_eq_of_le (hp₀ i hi)
  /-
    ι : Type u_1
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set ι
    p : ι → Submodule R M
    hp : ∀ (i : ι), Membership.mem s i → IsSemisimpleModule R (Subtype fun x => Me …
    q : Submodule R M := iSup fun i => iSup fun h => p i
    p' : ι → Submodule R (Subtype fun x => Membership.mem q x) := fun i => Submodu …
    hp₀ : ∀ (i : ι), Membership.mem s i → LE.le (p i) (LinearMap.range q.subtype)
    hp₁ : ∀ (i : ι), Membership.mem s i → IsSemisimpleModule R (Subtype fun x => M …
    hp₂ : Eq (iSup fun i => iSup fun h => p' i) Top.top
    ⊢ IsSemisimpleModule R (Subtype fun x => Membership.mem (iSup fun i => iSup fu …
  -/
  exact isSemisimpleModule_of_isSemisimpleModule_submodule hp₁ hp₂
  /-
    🎉 no goals
  -/


lemma isSemisimpleModule_of_isSemisimpleModule_submodule' {p : ι → Submodule R M}
    (hp : ∀ i, IsSemisimpleModule R (p i)) (hp' : ⨆ i, p i = ⊤) :
    IsSemisimpleModule R M :=
                                                                                          /-
                                                                                            ι : Type u_1
                                                                                            R : Type u_2
                                                                                            inst✝² : Ring R
                                                                                            M : Type u_4
                                                                                            inst✝¹ : AddCommGroup M
                                                                                            inst✝ : Module R M
                                                                                            p : ι → Submodule R M
                                                                                            hp : ∀ (i : ι), IsSemisimpleModule R (Subtype fun x => Membership.mem (p i) x)
                                                                                            hp' : Eq (iSup fun i => p i) Top.top
                                                                                            ⊢ Eq (iSup fun i => iSup fun h => p i) Top.top
                                                                                          -/
  isSemisimpleModule_of_isSemisimpleModule_submodule (s := Set.univ) (fun i _ ↦ hp i) (by simpa)
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem IsSemisimpleModule.sup {p q : Submodule R M}
    (_ : IsSemisimpleModule R p) (_ : IsSemisimpleModule R q) :
    IsSemisimpleModule R ↥(p ⊔ q) := by
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p q : Submodule R M
    x✝¹ : IsSemisimpleModule R (Subtype fun x => Membership.mem p x)
    x✝ : IsSemisimpleModule R (Subtype fun x => Membership.mem q x)
    ⊢ IsSemisimpleModule R (Subtype fun x => Membership.mem (Max.max p q) x)
  -/
  let f : Bool → Submodule R M := Bool.rec q p
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p q : Submodule R M
    x✝¹ : IsSemisimpleModule R (Subtype fun x => Membership.mem p x)
    x✝ : IsSemisimpleModule R (Subtype fun x => Membership.mem q x)
    f : Bool → Submodule R M := fun t => Bool.rec q p t
    ⊢ IsSemisimpleModule R (Subtype fun x => Membership.mem (Max.max p q) x)
  -/
  rw [show p ⊔ q = ⨆ i ∈ Set.univ, f i by rw [iSup_univ, iSup_bool_eq]]
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p q : Submodule R M
    x✝¹ : IsSemisimpleModule R (Subtype fun x => Membership.mem p x)
    x✝ : IsSemisimpleModule R (Subtype fun x => Membership.mem q x)
    f : Bool → Submodule R M := fun t => Bool.rec q p t
    ⊢ IsSemisimpleModule R (Subtype fun x => Membership.mem (iSup fun i => iSup fu …
  -/
  exact isSemisimpleModule_biSup_of_isSemisimpleModule_submodule (by rintro (_|_) _ <;> assumption)
  /-
    🎉 no goals
  -/


instance IsSemisimpleRing.isSemisimpleModule [IsSemisimpleRing R] : IsSemisimpleModule R M :=
  have : IsSemisimpleModule R (M →₀ R) := isSemisimpleModule_of_isSemisimpleModule_submodule'
    (fun _ ↦ .congr (LinearMap.quotKerEquivRange _).symm) Finsupp.iSup_lsingle_range
  .congr (LinearMap.quotKerEquivOfSurjective _ <| Finsupp.linearCombination_id_surjective R M).symm


instance IsSemisimpleRing.isCoatomic_submodule [IsSemisimpleRing R] : IsCoatomic (Submodule R M) :=
  isCoatomic_of_isAtomic_of_complementedLattice_of_isModular


open LinearMap in
/-- A finite product of semisimple rings is semisimple. -/
instance {ι} [Finite ι] (R : ι → Type*) [∀ i, Ring (R i)] [∀ i, IsSemisimpleRing (R i)] :
    IsSemisimpleRing (∀ i, R i) := by
  /-
    ι✝ : Type u_1
    R✝ : Type u_2
    S : Type u_3
    inst✝⁸ : Ring R✝
    inst✝⁷ : Ring S
    M : Type u_4
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R✝ M
    m : Submodule R✝ M
    N : Type u_5
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R✝ N
    ι : Type u_7
    inst✝² : Finite ι
    R : ι → Type u_6
    inst✝¹ : (i : ι) → Ring (R i)
    inst✝ : ∀ (i : ι), IsSemisimpleRing (R i)
    ⊢ IsSemisimpleRing ((i : ι) → R i)
  -/
  letI (i) : Module (∀ i, R i) (R i) := Module.compHom _ (Pi.evalRingHom R i)
  let e (i) : R i →ₛₗ[Pi.evalRingHom R i] R i :=
    { AddMonoidHom.id (R i) with map_smul' := fun _ _ ↦ rfl }
  have (i) : IsSemisimpleModule (∀ i, R i) (R i) :=
    ((e i).isSemisimpleModule_iff_of_bijective Function.bijective_id).mpr inferInstance
  classical
  exact isSemisimpleModule_of_isSemisimpleModule_submodule' (p := (range <| single _ _ ·))
    (fun i ↦ .range _) (by simp_rw [range_eq_map, Submodule.iSup_map_single, Submodule.pi_top])


/-- A binary product of semisimple rings is semisimple. -/
instance [hR : IsSemisimpleRing R] [hS : IsSemisimpleRing S] : IsSemisimpleRing (R × S) := by
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁵ : Ring R
    inst✝⁴ : Ring S
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    m : Submodule R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hR : IsSemisimpleRing R
    hS : IsSemisimpleRing S
    ⊢ IsSemisimpleRing (Prod R S)
  -/
  letI : Module (R × S) R := Module.compHom _ (.fst R S)
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁵ : Ring R
    inst✝⁴ : Ring S
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    m : Submodule R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hR : IsSemisimpleRing R
    hS : IsSemisimpleRing S
    this : Module (Prod R S) R := Module.compHom R (RingHom.fst R S)
    ⊢ IsSemisimpleRing (Prod R S)
  -/
  letI : Module (R × S) S := Module.compHom _ (.snd R S)
  -- e₁, e₂ got falsely flagged by the unused argument linter
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁵ : Ring R
    inst✝⁴ : Ring S
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    m : Submodule R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hR : IsSemisimpleRing R
    hS : IsSemisimpleRing S
    this✝ : Module (Prod R S) R := Module.compHom R (RingHom.fst R S)
    this : Module (Prod R S) S := Module.compHom S (RingHom.snd R S)
    ⊢ IsSemisimpleRing (Prod R S)
  -/
  let _e₁ : R →ₛₗ[.fst R S] R := { AddMonoidHom.id R with map_smul' := fun _ _ ↦ rfl }
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁵ : Ring R
    inst✝⁴ : Ring S
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    m : Submodule R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hR : IsSemisimpleRing R
    hS : IsSemisimpleRing S
    this✝ : Module (Prod R S) R := Module.compHom R (RingHom.fst R S)
    this : Module (Prod R S) S := Module.compHom S (RingHom.snd R S)
    _e₁ : LinearMap (RingHom.fst R S) R R :=
      let __src := AddMonoidHom.id R;
      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ }
    ⊢ IsSemisimpleRing (Prod R S)
  -/
  let _e₂ : S →ₛₗ[.snd R S] S := { AddMonoidHom.id S with map_smul' := fun _ _ ↦ rfl }
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁵ : Ring R
    inst✝⁴ : Ring S
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    m : Submodule R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hR : IsSemisimpleRing R
    hS : IsSemisimpleRing S
    this✝ : Module (Prod R S) R := Module.compHom R (RingHom.fst R S)
    this : Module (Prod R S) S := Module.compHom S (RingHom.snd R S)
    _e₁ : LinearMap (RingHom.fst R S) R R :=
      let __src := AddMonoidHom.id R;
      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ }
    _e₂ : LinearMap (RingHom.snd R S) S S :=
      let __src := AddMonoidHom.id S;
      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ }
    ⊢ IsSemisimpleRing (Prod R S)
  -/
  rw [IsSemisimpleRing, ← _e₁.isSemisimpleModule_iff_of_bijective Function.bijective_id] at hR
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁵ : Ring R
    inst✝⁴ : Ring S
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    m : Submodule R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hS : IsSemisimpleRing S
    this✝ : Module (Prod R S) R := Module.compHom R (RingHom.fst R S)
    hR : IsSemisimpleModule (Prod R S) R
    this : Module (Prod R S) S := Module.compHom S (RingHom.snd R S)
    _e₁ : LinearMap (RingHom.fst R S) R R :=
      let __src := AddMonoidHom.id R;
      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ }
    _e₂ : LinearMap (RingHom.snd R S) S S :=
      let __src := AddMonoidHom.id S;
      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ }
    ⊢ IsSemisimpleRing (Prod R S)
  -/
  rw [IsSemisimpleRing, ← _e₂.isSemisimpleModule_iff_of_bijective Function.bijective_id] at hS
  rw [IsSemisimpleRing, ← Submodule.topEquiv.isSemisimpleModule_iff_of_bijective
    (LinearEquiv.bijective _), ← LinearMap.sup_range_inl_inr]
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁵ : Ring R
    inst✝⁴ : Ring S
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    m : Submodule R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    this✝ : Module (Prod R S) R := Module.compHom R (RingHom.fst R S)
    hR : IsSemisimpleModule (Prod R S) R
    this : Module (Prod R S) S := Module.compHom S (RingHom.snd R S)
    hS : IsSemisimpleModule (Prod R S) S
    _e₁ : LinearMap (RingHom.fst R S) R R :=
      let __src := AddMonoidHom.id R;
      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ }
    _e₂ : LinearMap (RingHom.snd R S) S S :=
      let __src := AddMonoidHom.id S;
      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ }
    ⊢ IsSemisimpleModule (Prod R S) (Subtype fun x => Membership.mem (Max.max (Lin …
  -/
  exact .sup (.range _) (.range _)
  /-
    🎉 no goals
  -/


theorem RingHom.isSemisimpleRing_of_surjective (f : R →+* S) (hf : Function.Surjective f)
    [IsSemisimpleRing R] : IsSemisimpleRing S := by
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : Ring R
    inst✝¹ : Ring S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    inst✝ : IsSemisimpleRing R
    ⊢ IsSemisimpleRing S
  -/
  letI : Module R S := Module.compHom _ f
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : Ring R
    inst✝¹ : Ring S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    inst✝ : IsSemisimpleRing R
    this : Module R S := Module.compHom S f
    ⊢ IsSemisimpleRing S
  -/
  haveI : RingHomSurjective f := ⟨hf⟩
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : Ring R
    inst✝¹ : Ring S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    inst✝ : IsSemisimpleRing R
    this✝ : Module R S := Module.compHom S f
    this : RingHomSurjective f
    ⊢ IsSemisimpleRing S
  -/
  let e : S →ₛₗ[f] S := { AddMonoidHom.id S with map_smul' := fun _ _ ↦ rfl }
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : Ring R
    inst✝¹ : Ring S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    inst✝ : IsSemisimpleRing R
    this✝ : Module R S := Module.compHom S f
    this : RingHomSurjective f
    e : LinearMap f S S :=
      let __src := AddMonoidHom.id S;
      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ }
    ⊢ IsSemisimpleRing S
  -/
  rw [IsSemisimpleRing, ← e.isSemisimpleModule_iff_of_bijective Function.bijective_id]
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : Ring R
    inst✝¹ : Ring S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    inst✝ : IsSemisimpleRing R
    this✝ : Module R S := Module.compHom S f
    this : RingHomSurjective f
    e : LinearMap f S S :=
      let __src := AddMonoidHom.id S;
      { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ }
    ⊢ IsSemisimpleModule R S
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem IsSemisimpleRing.ideal_eq_span_idempotent [IsSemisimpleRing R] (I : Ideal R) :
    ∃ e : R, IsIdempotentElem e ∧ I = .span {e} := by
  /-
    R : Type u_2
    inst✝¹ : Ring R
    inst✝ : IsSemisimpleRing R
    I : Ideal R
    ⊢ Exists fun e => And (IsIdempotentElem e) (Eq I (Ideal.span (Singleton.single …
  -/
  obtain ⟨J, h⟩ := exists_isCompl I
  /-
    case intro
    R : Type u_2
    inst✝¹ : Ring R
    inst✝ : IsSemisimpleRing R
    I J : Ideal R
    h : IsCompl I J
    ⊢ Exists fun e => And (IsIdempotentElem e) (Eq I (Ideal.span (Singleton.single …
  -/
  obtain ⟨f, idem, rfl⟩ := I.isIdempotentElemEquiv.symm (I.isComplEquivProj ⟨J, h⟩)
  exact ⟨f 1, LinearMap.isIdempotentElem_apply_one_iff.mpr idem, by
    erw [LinearMap.range_eq_map, ← Ideal.span_one, LinearMap.map_span, Set.image_singleton]; rfl⟩


instance [IsSemisimpleRing R] : IsPrincipalIdealRing R where
  principal I := have ⟨e, _, he⟩ := IsSemisimpleRing.ideal_eq_span_idempotent I; ⟨e, he⟩


theorem injective_or_eq_zero [IsSimpleModule R M] (f : M →ₗ[R] N) :
    Function.Injective f ∨ f = 0 := by
  /-
    R : Type u_2
    inst✝⁵ : Ring R
    M : Type u_4
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u_5
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsSimpleModule R M
    f : LinearMap (RingHom.id R) M N
    ⊢ Or (Function.Injective ⇑f) (Eq f 0)
  -/
  rw [← ker_eq_bot, ← ker_eq_top]
  /-
    R : Type u_2
    inst✝⁵ : Ring R
    M : Type u_4
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u_5
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsSimpleModule R M
    f : LinearMap (RingHom.id R) M N
    ⊢ Or (Eq (LinearMap.ker f) Bot.bot) (Eq (LinearMap.ker f) Top.top)
  -/
  apply eq_bot_or_eq_top
  /-
    🎉 no goals
  -/


theorem injective_of_ne_zero [IsSimpleModule R M] {f : M →ₗ[R] N} (h : f ≠ 0) :
    Function.Injective f :=
  f.injective_or_eq_zero.resolve_right h


theorem surjective_or_eq_zero [IsSimpleModule R N] (f : M →ₗ[R] N) :
    Function.Surjective f ∨ f = 0 := by
  /-
    R : Type u_2
    inst✝⁵ : Ring R
    M : Type u_4
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u_5
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsSimpleModule R N
    f : LinearMap (RingHom.id R) M N
    ⊢ Or (Function.Surjective ⇑f) (Eq f 0)
  -/
  rw [← range_eq_top, ← range_eq_bot, or_comm]
  /-
    R : Type u_2
    inst✝⁵ : Ring R
    M : Type u_4
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u_5
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsSimpleModule R N
    f : LinearMap (RingHom.id R) M N
    ⊢ Or (Eq (LinearMap.range f) Bot.bot) (Eq (LinearMap.range f) Top.top)
  -/
  apply eq_bot_or_eq_top
  /-
    🎉 no goals
  -/


theorem surjective_of_ne_zero [IsSimpleModule R N] {f : M →ₗ[R] N} (h : f ≠ 0) :
    Function.Surjective f :=
  f.surjective_or_eq_zero.resolve_right h


/-- **Schur's Lemma** for linear maps between (possibly distinct) simple modules -/
theorem bijective_or_eq_zero [IsSimpleModule R M] [IsSimpleModule R N] (f : M →ₗ[R] N) :
    Function.Bijective f ∨ f = 0 :=
  or_iff_not_imp_right.mpr fun h ↦ ⟨injective_of_ne_zero h, surjective_of_ne_zero h⟩


theorem bijective_of_ne_zero [IsSimpleModule R M] [IsSimpleModule R N] {f : M →ₗ[R] N} (h : f ≠ 0) :
    Function.Bijective f :=
  f.bijective_or_eq_zero.resolve_right h


theorem isCoatom_ker_of_surjective [IsSimpleModule R N] {f : M →ₗ[R] N}
    (hf : Function.Surjective f) : IsCoatom (LinearMap.ker f) := by
  /-
    R : Type u_2
    inst✝⁵ : Ring R
    M : Type u_4
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u_5
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsSimpleModule R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Surjective ⇑f
    ⊢ IsCoatom (LinearMap.ker f)
  -/
  rw [← isSimpleModule_iff_isCoatom]
  /-
    R : Type u_2
    inst✝⁵ : Ring R
    M : Type u_4
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u_5
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : IsSimpleModule R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Surjective ⇑f
    ⊢ IsSimpleModule R (HasQuotient.Quotient M (LinearMap.ker f))
  -/
  exact IsSimpleModule.congr (f.quotKerEquivOfSurjective hf)
  /-
    🎉 no goals
  -/


/-- Schur's Lemma makes the endomorphism ring of a simple module a division ring. -/
noncomputable instance _root_.Module.End.divisionRing
    [DecidableEq (Module.End R M)] [IsSimpleModule R M] : DivisionRing (Module.End R M) where
  __ := Module.End.ring
  inv f := if h : f = 0 then 0 else (LinearEquiv.ofBijective _ <| bijective_of_ne_zero h).symm
  exists_pair_ne := ⟨0, 1, have := IsSimpleModule.nontrivial R M; zero_ne_one⟩
  mul_inv_cancel a a0 := by
    /-
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝⁷ : Ring R
      inst✝⁶ : Ring S
      M : Type u_4
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      m : Submodule R M
      N : Type u_5
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : DecidableEq (Module.End R M)
      inst✝ : IsSimpleModule R M
      a : Module.End R M
      a0 : Ne a 0
      ⊢ Eq (HMul.hMul a (Inv.inv a)) 1
    -/
    simp_rw [dif_neg a0]; ext
    /-
      case h
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      inst✝⁷ : Ring R
      inst✝⁶ : Ring S
      M : Type u_4
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      m : Submodule R M
      N : Type u_5
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : DecidableEq (Module.End R M)
      inst✝ : IsSimpleModule R M
      a : Module.End R M
      a0 : Ne a 0
      x✝ : M
      ⊢ Eq ((HMul.hMul a ↑(LinearEquiv.ofBijective a ⋯).symm) x✝) (1 x✝)
    -/
    exact (LinearEquiv.ofBijective _ <| bijective_of_ne_zero a0).right_inv _
    /-
      🎉 no goals
    -/
  inv_zero := dif_pos rfl
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


/-- An isomorphism `X₂ / X₁ ∩ X₂ ≅ Y₂ / Y₁ ∩ Y₂` of modules for pairs
`(X₁,X₂) (Y₁,Y₂) : Submodule R M` -/
def Iso (X Y : Submodule R M × Submodule R M) : Prop :=
  Nonempty <| (X.2 ⧸ X.1.comap X.2.subtype) ≃ₗ[R] Y.2 ⧸ Y.1.comap Y.2.subtype


theorem iso_symm {X Y : Submodule R M × Submodule R M} : Iso X Y → Iso Y X :=
  fun ⟨f⟩ => ⟨f.symm⟩


theorem iso_trans {X Y Z : Submodule R M × Submodule R M} : Iso X Y → Iso Y Z → Iso X Z :=
  fun ⟨f⟩ ⟨g⟩ => ⟨f.trans g⟩


@[nolint unusedArguments]
theorem second_iso {X Y : Submodule R M} (_ : X ⋖ X ⊔ Y) :
    Iso (X,X ⊔ Y) (X ⊓ Y,Y) := by
  /-
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    X Y : Submodule R M
    x✝ : CovBy X (Max.max X Y)
    ⊢ JordanHolderModule.Iso { fst := X, snd := Max.max X Y } { fst := Min.min X Y …
  -/
  constructor
  /-
    case val
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    X Y : Submodule R M
    x✝ : CovBy X (Max.max X Y)
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Membershi …
  -/
  rw [sup_comm, inf_comm]
  /-
    case val
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    X Y : Submodule R M
    x✝ : CovBy X (Max.max X Y)
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Membershi …
  -/
  dsimp
  /-
    case val
    R : Type u_2
    inst✝² : Ring R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    X Y : Submodule R M
    x✝ : CovBy X (Max.max X Y)
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient (Subtype fun x => Membershi …
  -/
  exact (LinearMap.quotientInfEquivSupQuotient Y X).symm
  /-
    🎉 no goals
  -/


instance instJordanHolderLattice : JordanHolderLattice (Submodule R M) where
  IsMaximal := (· ⋖ ·)
  lt_of_isMaximal := CovBy.lt
  sup_eq_of_isMaximal hxz hyz := WCovBy.sup_eq hxz.wcovBy hyz.wcovBy
  isMaximal_inf_left_of_isMaximal_sup := inf_covBy_of_covBy_sup_of_covBy_sup_left
  Iso := Iso
  iso_symm := iso_symm
  iso_trans := iso_trans
  second_iso := second_iso


