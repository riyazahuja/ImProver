theorem map_comap_le : map f (comap f N₂) ≤ N₂ :=
  (N₂ : Set M₂).image_preimage_subset f


theorem map_comap_eq (hf : N₂ ≤ f.range) : map f (comap f N₂) = N₂ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    M₂ : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₂
    inst✝ : LieRingModule L M₂
    N₂ : LieSubmodule R L M₂
    f : LieModuleHom R L M M₂
    hf : LE.le N₂ f.range
    ⊢ Eq (LieSubmodule.map f (LieSubmodule.comap f N₂)) N₂
  -/
  rw [SetLike.ext'_iff]
  /-
    R : Type u
    L : Type v
    M : Type w
    M₂ : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₂
    inst✝ : LieRingModule L M₂
    N₂ : LieSubmodule R L M₂
    f : LieModuleHom R L M M₂
    hf : LE.le N₂ f.range
    ⊢ Eq ↑(LieSubmodule.map f (LieSubmodule.comap f N₂)) ↑N₂
  -/
  exact Set.image_preimage_eq_of_subset hf
  /-
    🎉 no goals
  -/


theorem le_comap_map : N ≤ comap f (map f N) :=
  (N : Set M).subset_preimage_image f


theorem comap_map_eq (hf : f.ker = ⊥) : comap f (map f N) = N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    M₂ : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₂
    inst✝ : LieRingModule L M₂
    N : LieSubmodule R L M
    f : LieModuleHom R L M M₂
    hf : Eq f.ker Bot.bot
    ⊢ Eq (LieSubmodule.comap f (LieSubmodule.map f N)) N
  -/
  rw [SetLike.ext'_iff]
  /-
    R : Type u
    L : Type v
    M : Type w
    M₂ : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₂
    inst✝ : LieRingModule L M₂
    N : LieSubmodule R L M
    f : LieModuleHom R L M M₂
    hf : Eq f.ker Bot.bot
    ⊢ Eq ↑(LieSubmodule.comap f (LieSubmodule.map f N)) ↑N
  -/
  exact (N : Set M).preimage_image_eq (f.ker_eq_bot.mp hf)
  /-
    🎉 no goals
  -/


@[simp]
theorem map_comap_incl : map N.incl (comap N.incl N') = N ⊓ N' := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N N' : LieSubmodule R L M
    ⊢ Eq (LieSubmodule.map N.incl (LieSubmodule.comap N.incl N')) (Min.min N N')
  -/
  rw [← toSubmodule_inj]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N N' : LieSubmodule R L M
    ⊢ Eq ↑(LieSubmodule.map N.incl (LieSubmodule.comap N.incl N')) ↑(Min.min N N')
  -/
  exact (N : Submodule R M).map_comap_subtype N'
  /-
    🎉 no goals
  -/


/-- Given a Lie module `M` over a Lie algebra `L`, the set of Lie ideals of `L` acts on the set
of submodules of `M`. -/
instance hasBracket : Bracket (LieIdeal R L) (LieSubmodule R L M) :=
  ⟨fun I N => lieSpan R L { m | ∃ (x : I) (n : N), ⁅(x : L), (n : M)⁆ = m }⟩


theorem lieIdeal_oper_eq_span :
    ⁅I, N⁆ = lieSpan R L { m | ∃ (x : I) (n : N), ⁅(x : L), (n : M)⁆ = m } :=
  rfl


/-- See also `LieSubmodule.lieIdeal_oper_eq_linear_span'` and
`LieSubmodule.lieIdeal_oper_eq_tensor_map_range`. -/
theorem lieIdeal_oper_eq_linear_span [LieModule R L M] :
    (↑⁅I, N⁆ : Submodule R M) =
      Submodule.span R { m | ∃ (x : I) (n : N), ⁅(x : L), (n : M)⁆ = m } := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    N : LieSubmodule R L M
    inst✝¹ : LieAlgebra R L
    I : LieIdeal R L
    inst✝ : LieModule R L M
    ⊢ Eq (↑(Bracket.bracket I N)) (Submodule.span R (setOf fun m => Exists fun x = …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      I : LieIdeal R L
      inst✝ : LieModule R L M
      ⊢ LE.le (↑(Bracket.bracket I N)) (Submodule.span R (setOf fun m => Exists fun  …
    -/
  · let s := { m : M | ∃ (x : ↥I) (n : ↥N), ⁅(x : L), (n : M)⁆ = m }
    have aux : ∀ (y : L), ∀ m' ∈ Submodule.span R s, ⁅y, m'⁆ ∈ Submodule.span R s := by
      intro y m' hm'
      refine Submodule.span_induction (R := R) (M := M) (s := s)
        (p := fun m' _ ↦ ⁅y, m'⁆ ∈ Submodule.span R s) ?_ ?_ ?_ ?_ hm'
      · rintro m'' ⟨x, n, hm''⟩; rw [← hm'', leibniz_lie]
        refine Submodule.add_mem _ ?_ ?_ <;> apply Submodule.subset_span
        · use ⟨⁅y, ↑x⁆, I.lie_mem x.property⟩, n
        · use x, ⟨⁅y, ↑n⁆, N.lie_mem n.property⟩
      · simp only [lie_zero, Submodule.zero_mem]
      · intro m₁ m₂ _ _ hm₁ hm₂; rw [lie_add]; exact Submodule.add_mem _ hm₁ hm₂
      · intro t m'' _ hm''; rw [lie_smul]; exact Submodule.smul_mem _ t hm''
    /-
      case a
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      I : LieIdeal R L
      inst✝ : LieModule R L M
      s : Set M := setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.bracke …
      aux : ∀ (y : L) (m' : M), Membership.mem (Submodule.span R s) m' → Membership. …
      ⊢ LE.le (↑(Bracket.bracket I N)) (Submodule.span R (setOf fun m => Exists fun  …
    -/
    change _ ≤ ({ Submodule.span R s with lie_mem := fun hm' => aux _ _ hm' } : LieSubmodule R L M)
    /-
      case a
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      I : LieIdeal R L
      inst✝ : LieModule R L M
      s : Set M := setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.bracke …
      aux : ∀ (y : L) (m' : M), Membership.mem (Submodule.span R s) m' → Membership. …
      ⊢ LE.le (Bracket.bracket I N)
          (let __src := Submodule.span R s;
          { toSubmodule := __src, lie_mem := ⋯ })
    -/
    rw [lieIdeal_oper_eq_span, lieSpan_le]
    /-
      case a
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      I : LieIdeal R L
      inst✝ : LieModule R L M
      s : Set M := setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.bracke …
      aux : ∀ (y : L) (m' : M), Membership.mem (Submodule.span R s) m' → Membership. …
      ⊢ HasSubset.Subset (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket …
          ↑(let __src := Submodule.span R s;
            { toSubmodule := __src, lie_mem := ⋯ })
    -/
    exact Submodule.subset_span
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      I : LieIdeal R L
      inst✝ : LieModule R L M
      ⊢ LE.le (Submodule.span R (setOf fun m => Exists fun x => Exists fun n => Eq ( …
    -/
  · rw [lieIdeal_oper_eq_span]; apply submodule_span_le_lieSpan
                                /-
                                  🎉 no goals
                                -/


theorem lieIdeal_oper_eq_linear_span' [LieModule R L M] :
    (↑⁅I, N⁆ : Submodule R M) = Submodule.span R { m | ∃ x ∈ I, ∃ n ∈ N, ⁅x, n⁆ = m } := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    N : LieSubmodule R L M
    inst✝¹ : LieAlgebra R L
    I : LieIdeal R L
    inst✝ : LieModule R L M
    ⊢ Eq (↑(Bracket.bracket I N)) (Submodule.span R (setOf fun m => Exists fun x = …
  -/
  rw [lieIdeal_oper_eq_linear_span]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    N : LieSubmodule R L M
    inst✝¹ : LieAlgebra R L
    I : LieIdeal R L
    inst✝ : LieModule R L M
    ⊢ Eq (Submodule.span R (setOf fun m => Exists fun x => Exists fun n => Eq (Bra …
  -/
  congr
  /-
    case e_s
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    N : LieSubmodule R L M
    inst✝¹ : LieAlgebra R L
    I : LieIdeal R L
    inst✝ : LieModule R L M
    ⊢ Eq (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.bracket ↑x ↑n …
  -/
  ext m
  /-
    case e_s.h
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    N : LieSubmodule R L M
    inst✝¹ : LieAlgebra R L
    I : LieIdeal R L
    inst✝ : LieModule R L M
    m : M
    ⊢ Iff (Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq (Brac …
  -/
  constructor
    /-
      case e_s.h.mp
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      I : LieIdeal R L
      inst✝ : LieModule R L M
      m : M
      ⊢ Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.b …
    -/
  · rintro ⟨⟨x, hx⟩, ⟨n, hn⟩, rfl⟩
    /-
      case e_s.h.mp.intro.mk.intro.mk
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      I : LieIdeal R L
      inst✝ : LieModule R L M
      x : L
      hx : Membership.mem I x
      n : M
      hn : Membership.mem N n
      ⊢ Membership.mem (setOf fun m => Exists fun x => And (Membership.mem I x) (Exi …
    -/
    exact ⟨x, hx, n, hn, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case e_s.h.mpr
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      I : LieIdeal R L
      inst✝ : LieModule R L M
      m : M
      ⊢ Membership.mem (setOf fun m => Exists fun x => And (Membership.mem I x) (Exi …
    -/
  · rintro ⟨x, hx, n, hn, rfl⟩
    /-
      case e_s.h.mpr.intro.intro.intro.intro
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      I : LieIdeal R L
      inst✝ : LieModule R L M
      x : L
      hx : Membership.mem I x
      n : M
      hn : Membership.mem N n
      ⊢ Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.b …
    -/
    exact ⟨⟨x, hx⟩, ⟨n, hn⟩, rfl⟩
    /-
      🎉 no goals
    -/


theorem lie_le_iff : ⁅I, N⁆ ≤ N' ↔ ∀ x ∈ I, ∀ m ∈ N, ⁅x, m⁆ ∈ N' := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ Iff (LE.le (Bracket.bracket I N) N') (∀ (x : L), Membership.mem I x → ∀ (m : …
  -/
  rw [lieIdeal_oper_eq_span, LieSubmodule.lieSpan_le]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ Iff (HasSubset.Subset (setOf fun m => Exists fun x => Exists fun n => Eq (Br …
  -/
  refine ⟨fun h x hx m hm => h ⟨⟨x, hx⟩, ⟨m, hm⟩, rfl⟩, ?_⟩
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ (∀ (x : L), Membership.mem I x → ∀ (m : M), Membership.mem N m → Membership. …
  -/
  rintro h _ ⟨⟨x, hx⟩, ⟨m, hm⟩, rfl⟩
  /-
    case intro.mk.intro.mk
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h : ∀ (x : L), Membership.mem I x → ∀ (m : M), Membership.mem N m → Membership …
    x : L
    hx : Membership.mem I x
    m : M
    hm : Membership.mem N m
    ⊢ Membership.mem (↑N') (Bracket.bracket ↑⟨x, hx⟩ ↑⟨m, hm⟩)
  -/
  exact h x hx m hm
  /-
    🎉 no goals
  -/


variable {N I} in
theorem lie_coe_mem_lie (x : I) (m : N) : ⁅(x : L), (m : M)⁆ ∈ ⁅I, N⁆ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    x : Subtype fun x => Membership.mem I x
    m : Subtype fun x => Membership.mem N x
    ⊢ Membership.mem (Bracket.bracket I N) (Bracket.bracket ↑x ↑m)
  -/
  rw [lieIdeal_oper_eq_span]; apply subset_lieSpan; use x, m
                                                    /-
                                                      🎉 no goals
                                                    -/


variable {N I} in
theorem lie_mem_lie {x : L} {m : M} (hx : x ∈ I) (hm : m ∈ N) : ⁅x, m⁆ ∈ ⁅I, N⁆ :=
  lie_coe_mem_lie ⟨x, hx⟩ ⟨m, hm⟩


theorem lie_comm : ⁅I, J⁆ = ⁅J, I⁆ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    ⊢ Eq (Bracket.bracket I J) (Bracket.bracket J I)
  -/
  suffices ∀ I J : LieIdeal R L, ⁅I, J⁆ ≤ ⁅J, I⁆ by exact le_antisymm (this I J) (this J I)
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    ⊢ ∀ (I J : LieIdeal R L), LE.le (Bracket.bracket I J) (Bracket.bracket J I)
  -/
  clear! I J; intro I J
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    ⊢ LE.le (Bracket.bracket I J) (Bracket.bracket J I)
  -/
  rw [lieIdeal_oper_eq_span, lieSpan_le]; rintro x ⟨y, z, h⟩; rw [← h]
  /-
    case intro.intro
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    x : L
    y : Subtype fun x => Membership.mem I x
    z : Subtype fun x => Membership.mem J x
    h : Eq (Bracket.bracket ↑y ↑z) x
    ⊢ Membership.mem (↑(Bracket.bracket J I)) (Bracket.bracket ↑y ↑z)
  -/
  rw [← lie_skew, ← lie_neg, ← LieSubmodule.coe_neg]
  /-
    case intro.intro
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    x : L
    y : Subtype fun x => Membership.mem I x
    z : Subtype fun x => Membership.mem J x
    h : Eq (Bracket.bracket ↑y ↑z) x
    ⊢ Membership.mem (↑(Bracket.bracket J I)) (Bracket.bracket ↑z ↑(Neg.neg y))
  -/
  apply lie_coe_mem_lie
  /-
    🎉 no goals
  -/


theorem lie_le_right : ⁅I, N⁆ ≤ N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ LE.le (Bracket.bracket I N) N
  -/
  rw [lieIdeal_oper_eq_span, lieSpan_le]; rintro m ⟨x, n, hn⟩; rw [← hn]
  /-
    case intro.intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    m : M
    x : Subtype fun x => Membership.mem I x
    n : Subtype fun x => Membership.mem N x
    hn : Eq (Bracket.bracket ↑x ↑n) m
    ⊢ Membership.mem (↑N) (Bracket.bracket ↑x ↑n)
  -/
  exact N.lie_mem n.property
  /-
    🎉 no goals
  -/


                                       /-
                                         R : Type u
                                         L : Type v
                                         inst✝² : CommRing R
                                         inst✝¹ : LieRing L
                                         inst✝ : LieAlgebra R L
                                         I J : LieIdeal R L
                                         ⊢ LE.le (Bracket.bracket I J) I
                                       -/
theorem lie_le_left : ⁅I, J⁆ ≤ I := by rw [lie_comm]; exact lie_le_right I J
                                                      /-
                                                        🎉 no goals
                                                      -/


                                          /-
                                            R : Type u
                                            L : Type v
                                            inst✝² : CommRing R
                                            inst✝¹ : LieRing L
                                            inst✝ : LieAlgebra R L
                                            I J : LieIdeal R L
                                            ⊢ LE.le (Bracket.bracket I J) (Min.min I J)
                                          -/
theorem lie_le_inf : ⁅I, J⁆ ≤ I ⊓ J := by rw [le_inf_iff]; exact ⟨lie_le_left I J, lie_le_right J I⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                                          /-
                                                            R : Type u
                                                            L : Type v
                                                            M : Type w
                                                            inst✝⁵ : CommRing R
                                                            inst✝⁴ : LieRing L
                                                            inst✝³ : AddCommGroup M
                                                            inst✝² : Module R M
                                                            inst✝¹ : LieRingModule L M
                                                            inst✝ : LieAlgebra R L
                                                            I : LieIdeal R L
                                                            ⊢ Eq (Bracket.bracket I Bot.bot) Bot.bot
                                                          -/
theorem lie_bot : ⁅I, (⊥ : LieSubmodule R L M)⁆ = ⊥ := by rw [eq_bot_iff]; apply lie_le_right
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem bot_lie : ⁅(⊥ : LieIdeal R L), N⁆ = ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    ⊢ Eq (Bracket.bracket Bot.bot N) Bot.bot
  -/
  suffices ⁅(⊥ : LieIdeal R L), N⁆ ≤ ⊥ by exact le_bot_iff.mp this
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    ⊢ LE.le (Bracket.bracket Bot.bot N) Bot.bot
  -/
  rw [lieIdeal_oper_eq_span, lieSpan_le]; rintro m ⟨⟨x, hx⟩, n, hn⟩; rw [← hn]
  /-
    case intro.mk.intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    m : M
    x : L
    hx : Membership.mem Bot.bot x
    n : Subtype fun x => Membership.mem N x
    hn : Eq (Bracket.bracket ↑⟨x, hx⟩ ↑n) m
    ⊢ Membership.mem (↑Bot.bot) (Bracket.bracket ↑⟨x, hx⟩ ↑n)
  -/
  change x ∈ (⊥ : LieIdeal R L) at hx; rw [mem_bot] at hx; simp [hx]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem lie_eq_bot_iff : ⁅I, N⁆ = ⊥ ↔ ∀ x ∈ I, ∀ m ∈ N, ⁅(x : L), m⁆ = 0 := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ Iff (Eq (Bracket.bracket I N) Bot.bot) (∀ (x : L), Membership.mem I x → ∀ (m …
  -/
  rw [lieIdeal_oper_eq_span, LieSubmodule.lieSpan_eq_bot_iff]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ Iff (∀ (m : M), Membership.mem (setOf fun m => Exists fun x => Exists fun n  …
  -/
  refine ⟨fun h x hx m hm => h ⁅x, m⁆ ⟨⟨x, hx⟩, ⟨m, hm⟩, rfl⟩, ?_⟩
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ (∀ (x : L), Membership.mem I x → ∀ (m : M), Membership.mem N m → Eq (Bracket …
  -/
  rintro h - ⟨⟨x, hx⟩, ⟨⟨n, hn⟩, rfl⟩⟩
  /-
    case intro.mk.intro.mk
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h : ∀ (x : L), Membership.mem I x → ∀ (m : M), Membership.mem N m → Eq (Bracke …
    x : L
    hx : Membership.mem I x
    n : M
    hn : Membership.mem N n
    ⊢ Eq (Bracket.bracket ↑⟨x, hx⟩ ↑⟨n, hn⟩) 0
  -/
  exact h x hx n hn
  /-
    🎉 no goals
  -/


variable {I J N N'} in
theorem mono_lie (h₁ : I ≤ J) (h₂ : N ≤ N') : ⁅I, N⁆ ≤ ⁅J, N'⁆ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    h₁ : LE.le I J
    h₂ : LE.le N N'
    ⊢ LE.le (Bracket.bracket I N) (Bracket.bracket J N')
  -/
  intro m h
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    h₁ : LE.le I J
    h₂ : LE.le N N'
    m : M
    h : Membership.mem (Bracket.bracket I N) m
    ⊢ Membership.mem (Bracket.bracket J N') m
  -/
  rw [lieIdeal_oper_eq_span, mem_lieSpan] at h; rw [lieIdeal_oper_eq_span, mem_lieSpan]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    h₁ : LE.le I J
    h₂ : LE.le N N'
    m : M
    h : ∀ (N_1 : LieSubmodule R L M), HasSubset.Subset (setOf fun m => Exists fun  …
    ⊢ ∀ (N : LieSubmodule R L M), HasSubset.Subset (setOf fun m => Exists fun x => …
  -/
  intro N hN; apply h; rintro m' ⟨⟨x, hx⟩, ⟨n, hn⟩, hm⟩; rw [← hm]; apply hN
  /-
    case a.intro.mk.intro.mk.a
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N✝ N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    h₁ : LE.le I J
    h₂ : LE.le N✝ N'
    m : M
    h : ∀ (N : LieSubmodule R L M), HasSubset.Subset (setOf fun m => Exists fun x  …
    N : LieSubmodule R L M
    hN : HasSubset.Subset (setOf fun m => Exists fun x => Exists fun n => Eq (Brac …
    m' : M
    x : L
    hx : Membership.mem I x
    n : M
    hn : Membership.mem N✝ n
    hm : Eq (Bracket.bracket ↑⟨x, hx⟩ ↑⟨n, hn⟩) m'
    ⊢ Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.b …
  -/
  use ⟨x, h₁ hx⟩, ⟨n, h₂ hn⟩
  /-
    🎉 no goals
  -/


variable {I J} in
theorem mono_lie_left (h : I ≤ J) : ⁅I, N⁆ ≤ ⁅J, N⁆ :=
  mono_lie h (le_refl N)


variable {N N'} in
theorem mono_lie_right (h : N ≤ N') : ⁅I, N⁆ ≤ ⁅I, N'⁆ :=
  mono_lie (le_refl I) h


@[simp]
theorem lie_sup : ⁅I, N ⊔ N'⁆ = ⁅I, N⁆ ⊔ ⁅I, N'⁆ := by
  have h : ⁅I, N⁆ ⊔ ⁅I, N'⁆ ≤ ⁅I, N ⊔ N'⁆ := by
    rw [sup_le_iff]; constructor <;>
    apply mono_lie_right <;> [exact le_sup_left; exact le_sup_right]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket I N')) (Bracket.brac …
    ⊢ Eq (Bracket.bracket I (Max.max N N')) (Max.max (Bracket.bracket I N) (Bracke …
  -/
  suffices ⁅I, N ⊔ N'⁆ ≤ ⁅I, N⁆ ⊔ ⁅I, N'⁆ by exact le_antisymm this h
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket I N')) (Bracket.brac …
    ⊢ LE.le (Bracket.bracket I (Max.max N N')) (Max.max (Bracket.bracket I N) (Bra …
  -/
  rw [lieIdeal_oper_eq_span, lieSpan_le]; rintro m ⟨x, ⟨n, hn⟩, h⟩; erw [LieSubmodule.mem_sup]
  /-
    case intro.intro.mk
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h✝ : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket I N')) (Bracket.bra …
    m : M
    x : Subtype fun x => Membership.mem I x
    n : M
    hn : Membership.mem (Max.max N N') n
    h : Eq (Bracket.bracket ↑x ↑⟨n, hn⟩) m
    ⊢ Exists fun y => And (Membership.mem (Bracket.bracket I N) y) (Exists fun z = …
  -/
  rw [LieSubmodule.mem_sup] at hn; rcases hn with ⟨n₁, hn₁, n₂, hn₂, hn'⟩
  /-
    case intro.intro.mk.intro.intro.intro.intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h✝ : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket I N')) (Bracket.bra …
    m : M
    x : Subtype fun x => Membership.mem I x
    n : M
    hn : Membership.mem (Max.max N N') n
    h : Eq (Bracket.bracket ↑x ↑⟨n, hn⟩) m
    n₁ : M
    hn₁ : Membership.mem N n₁
    n₂ : M
    hn₂ : Membership.mem N' n₂
    hn' : Eq (HAdd.hAdd n₁ n₂) n
    ⊢ Exists fun y => And (Membership.mem (Bracket.bracket I N) y) (Exists fun z = …
  -/
  use ⁅(x : L), (⟨n₁, hn₁⟩ : N)⁆; constructor; · apply lie_coe_mem_lie
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    case h.right
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h✝ : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket I N')) (Bracket.bra …
    m : M
    x : Subtype fun x => Membership.mem I x
    n : M
    hn : Membership.mem (Max.max N N') n
    h : Eq (Bracket.bracket ↑x ↑⟨n, hn⟩) m
    n₁ : M
    hn₁ : Membership.mem N n₁
    n₂ : M
    hn₂ : Membership.mem N' n₂
    hn' : Eq (HAdd.hAdd n₁ n₂) n
    ⊢ Exists fun z => And (Membership.mem (Bracket.bracket I N') z) (Eq (HAdd.hAdd …
  -/
  use ⁅(x : L), (⟨n₂, hn₂⟩ : N')⁆; constructor; · apply lie_coe_mem_lie
                                                  /-
                                                    🎉 no goals
                                                  -/
  /-
    case h.right
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    h✝ : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket I N')) (Bracket.bra …
    m : M
    x : Subtype fun x => Membership.mem I x
    n : M
    hn : Membership.mem (Max.max N N') n
    h : Eq (Bracket.bracket ↑x ↑⟨n, hn⟩) m
    n₁ : M
    hn₁ : Membership.mem N n₁
    n₂ : M
    hn₂ : Membership.mem N' n₂
    hn' : Eq (HAdd.hAdd n₁ n₂) n
    ⊢ Eq (HAdd.hAdd (Bracket.bracket ↑x ↑⟨n₁, hn₁⟩) (Bracket.bracket ↑x ↑⟨n₂, hn₂⟩ …
  -/
  simp [← h, ← hn']
  /-
    🎉 no goals
  -/


@[simp]
theorem sup_lie : ⁅I ⊔ J, N⁆ = ⁅I, N⁆ ⊔ ⁅J, N⁆ := by
  have h : ⁅I, N⁆ ⊔ ⁅J, N⁆ ≤ ⁅I ⊔ J, N⁆ := by
    rw [sup_le_iff]; constructor <;>
    apply mono_lie_left <;> [exact le_sup_left; exact le_sup_right]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    h : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket J N)) (Bracket.brack …
    ⊢ Eq (Bracket.bracket (Max.max I J) N) (Max.max (Bracket.bracket I N) (Bracket …
  -/
  suffices ⁅I ⊔ J, N⁆ ≤ ⁅I, N⁆ ⊔ ⁅J, N⁆ by exact le_antisymm this h
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    h : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket J N)) (Bracket.brack …
    ⊢ LE.le (Bracket.bracket (Max.max I J) N) (Max.max (Bracket.bracket I N) (Brac …
  -/
  rw [lieIdeal_oper_eq_span, lieSpan_le]; rintro m ⟨⟨x, hx⟩, n, h⟩; erw [LieSubmodule.mem_sup]
  /-
    case intro.mk.intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    h✝ : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket J N)) (Bracket.brac …
    m : M
    x : L
    hx : Membership.mem (Max.max I J) x
    n : Subtype fun x => Membership.mem N x
    h : Eq (Bracket.bracket ↑⟨x, hx⟩ ↑n) m
    ⊢ Exists fun y => And (Membership.mem (Bracket.bracket I N) y) (Exists fun z = …
  -/
  rw [LieSubmodule.mem_sup] at hx; rcases hx with ⟨x₁, hx₁, x₂, hx₂, hx'⟩
  /-
    case intro.mk.intro.intro.intro.intro.intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    h✝ : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket J N)) (Bracket.brac …
    m : M
    x : L
    hx : Membership.mem (Max.max I J) x
    n : Subtype fun x => Membership.mem N x
    h : Eq (Bracket.bracket ↑⟨x, hx⟩ ↑n) m
    x₁ : L
    hx₁ : Membership.mem I x₁
    x₂ : L
    hx₂ : Membership.mem J x₂
    hx' : Eq (HAdd.hAdd x₁ x₂) x
    ⊢ Exists fun y => And (Membership.mem (Bracket.bracket I N) y) (Exists fun z = …
  -/
  use ⁅((⟨x₁, hx₁⟩ : I) : L), (n : N)⁆; constructor; · apply lie_coe_mem_lie
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    case h.right
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    h✝ : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket J N)) (Bracket.brac …
    m : M
    x : L
    hx : Membership.mem (Max.max I J) x
    n : Subtype fun x => Membership.mem N x
    h : Eq (Bracket.bracket ↑⟨x, hx⟩ ↑n) m
    x₁ : L
    hx₁ : Membership.mem I x₁
    x₂ : L
    hx₂ : Membership.mem J x₂
    hx' : Eq (HAdd.hAdd x₁ x₂) x
    ⊢ Exists fun z => And (Membership.mem (Bracket.bracket J N) z) (Eq (HAdd.hAdd  …
  -/
  use ⁅((⟨x₂, hx₂⟩ : J) : L), (n : N)⁆; constructor; · apply lie_coe_mem_lie
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    case h.right
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    h✝ : LE.le (Max.max (Bracket.bracket I N) (Bracket.bracket J N)) (Bracket.brac …
    m : M
    x : L
    hx : Membership.mem (Max.max I J) x
    n : Subtype fun x => Membership.mem N x
    h : Eq (Bracket.bracket ↑⟨x, hx⟩ ↑n) m
    x₁ : L
    hx₁ : Membership.mem I x₁
    x₂ : L
    hx₂ : Membership.mem J x₂
    hx' : Eq (HAdd.hAdd x₁ x₂) x
    ⊢ Eq (HAdd.hAdd (Bracket.bracket ↑⟨x₁, hx₁⟩ ↑n) (Bracket.bracket ↑⟨x₂, hx₂⟩ ↑n …
  -/
  simp [← h, ← hx']
  /-
    🎉 no goals
  -/


theorem lie_inf : ⁅I, N ⊓ N'⁆ ≤ ⁅I, N⁆ ⊓ ⁅I, N'⁆ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ LE.le (Bracket.bracket I (Min.min N N')) (Min.min (Bracket.bracket I N) (Bra …
  -/
  rw [le_inf_iff]; constructor <;>
  apply mono_lie_right <;> [exact inf_le_left; exact inf_le_right]


theorem inf_lie : ⁅I ⊓ J, N⁆ ≤ ⁅I, N⁆ ⊓ ⁅J, N⁆ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    inst✝ : LieAlgebra R L
    I J : LieIdeal R L
    ⊢ LE.le (Bracket.bracket (Min.min I J) N) (Min.min (Bracket.bracket I N) (Brac …
  -/
  rw [le_inf_iff]; constructor <;>
  apply mono_lie_left <;> [exact inf_le_left; exact inf_le_right]


theorem map_bracket_eq [LieModule R L M] : map f ⁅I, N⁆ = ⁅I, map f N⁆ := by
  rw [← toSubmodule_inj, toSubmodule_map, lieIdeal_oper_eq_linear_span,
    lieIdeal_oper_eq_linear_span, Submodule.map_span]
  /-
    R : Type u
    L : Type v
    M : Type w
    M₂ : Type w₁
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R M₂
    inst✝³ : LieRingModule L M₂
    N : LieSubmodule R L M
    f : LieModuleHom R L M M₂
    inst✝² : LieAlgebra R L
    inst✝¹ : LieModule R L M₂
    I : LieIdeal R L
    inst✝ : LieModule R L M
    ⊢ Eq (Submodule.span R (Set.image (⇑↑f) (setOf fun m => Exists fun x => Exists …
  -/
  congr
  /-
    case e_s
    R : Type u
    L : Type v
    M : Type w
    M₂ : Type w₁
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R M₂
    inst✝³ : LieRingModule L M₂
    N : LieSubmodule R L M
    f : LieModuleHom R L M M₂
    inst✝² : LieAlgebra R L
    inst✝¹ : LieModule R L M₂
    I : LieIdeal R L
    inst✝ : LieModule R L M
    ⊢ Eq (Set.image (⇑↑f) (setOf fun m => Exists fun x => Exists fun n => Eq (Brac …
  -/
  ext m
  /-
    case e_s.h
    R : Type u
    L : Type v
    M : Type w
    M₂ : Type w₁
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R M₂
    inst✝³ : LieRingModule L M₂
    N : LieSubmodule R L M
    f : LieModuleHom R L M M₂
    inst✝² : LieAlgebra R L
    inst✝¹ : LieModule R L M₂
    I : LieIdeal R L
    inst✝ : LieModule R L M
    m : M₂
    ⊢ Iff (Membership.mem (Set.image (⇑↑f) (setOf fun m => Exists fun x => Exists  …
  -/
  constructor
    /-
      case e_s.h.mp
      R : Type u
      L : Type v
      M : Type w
      M₂ : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M₂
      inst✝³ : LieRingModule L M₂
      N : LieSubmodule R L M
      f : LieModuleHom R L M M₂
      inst✝² : LieAlgebra R L
      inst✝¹ : LieModule R L M₂
      I : LieIdeal R L
      inst✝ : LieModule R L M
      m : M₂
      ⊢ Membership.mem (Set.image (⇑↑f) (setOf fun m => Exists fun x => Exists fun n …
    -/
  · rintro ⟨-, ⟨⟨x, ⟨n, hn⟩, rfl⟩, hm⟩⟩
    /-
      case e_s.h.mp.intro.intro.intro.intro.mk
      R : Type u
      L : Type v
      M : Type w
      M₂ : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M₂
      inst✝³ : LieRingModule L M₂
      N : LieSubmodule R L M
      f : LieModuleHom R L M M₂
      inst✝² : LieAlgebra R L
      inst✝¹ : LieModule R L M₂
      I : LieIdeal R L
      inst✝ : LieModule R L M
      m : M₂
      x : Subtype fun x => Membership.mem I x
      n : M
      hn : Membership.mem N n
      hm : Eq (↑f (Bracket.bracket ↑x ↑⟨n, hn⟩)) m
      ⊢ Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.b …
    -/
    simp only [LieModuleHom.coe_toLinearMap, LieModuleHom.map_lie] at hm
    /-
      case e_s.h.mp.intro.intro.intro.intro.mk
      R : Type u
      L : Type v
      M : Type w
      M₂ : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M₂
      inst✝³ : LieRingModule L M₂
      N : LieSubmodule R L M
      f : LieModuleHom R L M M₂
      inst✝² : LieAlgebra R L
      inst✝¹ : LieModule R L M₂
      I : LieIdeal R L
      inst✝ : LieModule R L M
      m : M₂
      x : Subtype fun x => Membership.mem I x
      n : M
      hn : Membership.mem N n
      hm : Eq (Bracket.bracket (↑x) (f n)) m
      ⊢ Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.b …
    -/
    exact ⟨x, ⟨f n, (mem_map (f n)).mpr ⟨n, hn, rfl⟩⟩, hm⟩
    /-
      🎉 no goals
    -/
    /-
      case e_s.h.mpr
      R : Type u
      L : Type v
      M : Type w
      M₂ : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M₂
      inst✝³ : LieRingModule L M₂
      N : LieSubmodule R L M
      f : LieModuleHom R L M M₂
      inst✝² : LieAlgebra R L
      inst✝¹ : LieModule R L M₂
      I : LieIdeal R L
      inst✝ : LieModule R L M
      m : M₂
      ⊢ Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.b …
    -/
  · rintro ⟨x, ⟨m₂, hm₂ : m₂ ∈ map f N⟩, rfl⟩
    /-
      case e_s.h.mpr.intro.intro.mk
      R : Type u
      L : Type v
      M : Type w
      M₂ : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M₂
      inst✝³ : LieRingModule L M₂
      N : LieSubmodule R L M
      f : LieModuleHom R L M M₂
      inst✝² : LieAlgebra R L
      inst✝¹ : LieModule R L M₂
      I : LieIdeal R L
      inst✝ : LieModule R L M
      x : Subtype fun x => Membership.mem I x
      m₂ : M₂
      hm₂ : Membership.mem (LieSubmodule.map f N) m₂
      ⊢ Membership.mem (Set.image (⇑↑f) (setOf fun m => Exists fun x => Exists fun n …
    -/
    obtain ⟨n, hn, rfl⟩ := (mem_map m₂).mp hm₂
    /-
      case e_s.h.mpr.intro.intro.mk.intro.intro
      R : Type u
      L : Type v
      M : Type w
      M₂ : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M₂
      inst✝³ : LieRingModule L M₂
      N : LieSubmodule R L M
      f : LieModuleHom R L M M₂
      inst✝² : LieAlgebra R L
      inst✝¹ : LieModule R L M₂
      I : LieIdeal R L
      inst✝ : LieModule R L M
      x : Subtype fun x => Membership.mem I x
      n : M
      hn : Membership.mem N n
      hm₂ : Membership.mem (LieSubmodule.map f N) (f n)
      ⊢ Membership.mem (Set.image (⇑↑f) (setOf fun m => Exists fun x => Exists fun n …
    -/
    exact ⟨⁅x, n⁆, ⟨x, ⟨n, hn⟩, rfl⟩, by simp⟩
    /-
      🎉 no goals
    -/


theorem comap_bracket_eq [LieModule R L M] (hf₁ : f.ker = ⊥) (hf₂ : N₂ ≤ f.range) :
    comap f ⁅I, N₂⁆ = ⁅I, comap f N₂⁆ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    M₂ : Type w₁
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R M₂
    inst✝³ : LieRingModule L M₂
    N₂ : LieSubmodule R L M₂
    f : LieModuleHom R L M M₂
    inst✝² : LieAlgebra R L
    inst✝¹ : LieModule R L M₂
    I : LieIdeal R L
    inst✝ : LieModule R L M
    hf₁ : Eq f.ker Bot.bot
    hf₂ : LE.le N₂ f.range
    ⊢ Eq (LieSubmodule.comap f (Bracket.bracket I N₂)) (Bracket.bracket I (LieSubm …
  -/
  conv_lhs => rw [← map_comap_eq N₂ f hf₂]
  /-
    R : Type u
    L : Type v
    M : Type w
    M₂ : Type w₁
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R M₂
    inst✝³ : LieRingModule L M₂
    N₂ : LieSubmodule R L M₂
    f : LieModuleHom R L M M₂
    inst✝² : LieAlgebra R L
    inst✝¹ : LieModule R L M₂
    I : LieIdeal R L
    inst✝ : LieModule R L M
    hf₁ : Eq f.ker Bot.bot
    hf₂ : LE.le N₂ f.range
    ⊢ Eq (LieSubmodule.comap f (Bracket.bracket I (LieSubmodule.map f (LieSubmodul …
  -/
  rw [← map_bracket_eq, comap_map_eq _ f hf₁]
  /-
    🎉 no goals
  -/


/-- Note that the inequality can be strict; e.g., the inclusion of an Abelian subalgebra of a
simple algebra. -/
theorem map_bracket_le {I₁ I₂ : LieIdeal R L} : map f ⁅I₁, I₂⁆ ≤ ⁅map f I₁, map f I₂⁆ := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    ⊢ LE.le (LieIdeal.map f (Bracket.bracket I₁ I₂)) (Bracket.bracket (LieIdeal.ma …
  -/
  rw [map_le_iff_le_comap]; erw [LieSubmodule.lieSpan_le]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    ⊢ HasSubset.Subset (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket …
  -/
  intro x hx; obtain ⟨⟨y₁, hy₁⟩, ⟨y₂, hy₂⟩, hx⟩ := hx; rw [← hx]
  /-
    case intro.mk.intro.mk
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    x y₁ : L
    hy₁ : Membership.mem I₁ y₁
    y₂ : L
    hy₂ : Membership.mem I₂ y₂
    hx : Eq (Bracket.bracket ↑⟨y₁, hy₁⟩ ↑⟨y₂, hy₂⟩) x
    ⊢ Membership.mem (↑(LieIdeal.comap f (Bracket.bracket (LieIdeal.map f I₁) (Lie …
  -/
  let fy₁ : ↥(map f I₁) := ⟨f y₁, mem_map hy₁⟩
  /-
    case intro.mk.intro.mk
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    x y₁ : L
    hy₁ : Membership.mem I₁ y₁
    y₂ : L
    hy₂ : Membership.mem I₂ y₂
    hx : Eq (Bracket.bracket ↑⟨y₁, hy₁⟩ ↑⟨y₂, hy₂⟩) x
    fy₁ : Subtype fun x => Membership.mem (LieIdeal.map f I₁) x := ⟨f y₁, ⋯⟩
    ⊢ Membership.mem (↑(LieIdeal.comap f (Bracket.bracket (LieIdeal.map f I₁) (Lie …
  -/
  let fy₂ : ↥(map f I₂) := ⟨f y₂, mem_map hy₂⟩
  /-
    case intro.mk.intro.mk
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    x y₁ : L
    hy₁ : Membership.mem I₁ y₁
    y₂ : L
    hy₂ : Membership.mem I₂ y₂
    hx : Eq (Bracket.bracket ↑⟨y₁, hy₁⟩ ↑⟨y₂, hy₂⟩) x
    fy₁ : Subtype fun x => Membership.mem (LieIdeal.map f I₁) x := ⟨f y₁, ⋯⟩
    fy₂ : Subtype fun x => Membership.mem (LieIdeal.map f I₂) x := ⟨f y₂, ⋯⟩
    ⊢ Membership.mem (↑(LieIdeal.comap f (Bracket.bracket (LieIdeal.map f I₁) (Lie …
  -/
  change _ ∈ comap f ⁅map f I₁, map f I₂⁆
  /-
    case intro.mk.intro.mk
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    x y₁ : L
    hy₁ : Membership.mem I₁ y₁
    y₂ : L
    hy₂ : Membership.mem I₂ y₂
    hx : Eq (Bracket.bracket ↑⟨y₁, hy₁⟩ ↑⟨y₂, hy₂⟩) x
    fy₁ : Subtype fun x => Membership.mem (LieIdeal.map f I₁) x := ⟨f y₁, ⋯⟩
    fy₂ : Subtype fun x => Membership.mem (LieIdeal.map f I₂) x := ⟨f y₂, ⋯⟩
    ⊢ Membership.mem (LieIdeal.comap f (Bracket.bracket (LieIdeal.map f I₁) (LieId …
  -/
  simp only [Submodule.coe_mk, mem_comap, LieHom.map_lie]
  /-
    case intro.mk.intro.mk
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    x y₁ : L
    hy₁ : Membership.mem I₁ y₁
    y₂ : L
    hy₂ : Membership.mem I₂ y₂
    hx : Eq (Bracket.bracket ↑⟨y₁, hy₁⟩ ↑⟨y₂, hy₂⟩) x
    fy₁ : Subtype fun x => Membership.mem (LieIdeal.map f I₁) x := ⟨f y₁, ⋯⟩
    fy₂ : Subtype fun x => Membership.mem (LieIdeal.map f I₂) x := ⟨f y₂, ⋯⟩
    ⊢ Membership.mem (Bracket.bracket (LieIdeal.map f I₁) (LieIdeal.map f I₂)) (Br …
  -/
  exact LieSubmodule.lie_coe_mem_lie fy₁ fy₂
  /-
    🎉 no goals
  -/


theorem map_bracket_eq {I₁ I₂ : LieIdeal R L} (h : Function.Surjective f) :
    map f ⁅I₁, I₂⁆ = ⁅map f I₁, map f I₂⁆ := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    h : Function.Surjective ⇑f
    ⊢ Eq (LieIdeal.map f (Bracket.bracket I₁ I₂)) (Bracket.bracket (LieIdeal.map f …
  -/
  suffices ⁅map f I₁, map f I₂⁆ ≤ map f ⁅I₁, I₂⁆ by exact le_antisymm (map_bracket_le f) this
  rw [← LieSubmodule.toSubmodule_le_toSubmodule, coe_map_of_surjective h,
    LieSubmodule.lieIdeal_oper_eq_linear_span, LieSubmodule.lieIdeal_oper_eq_linear_span,
    LinearMap.map_span]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    h : Function.Surjective ⇑f
    ⊢ LE.le (Submodule.span R (setOf fun m => Exists fun x => Exists fun n => Eq ( …
  -/
  apply Submodule.span_mono
  /-
    case h
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    h : Function.Surjective ⇑f
    ⊢ HasSubset.Subset (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket …
  -/
  rintro x ⟨⟨z₁, h₁⟩, ⟨z₂, h₂⟩, rfl⟩
  /-
    case h.intro.mk.intro.mk
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    h : Function.Surjective ⇑f
    z₁ : L'
    h₁ : Membership.mem (LieIdeal.map f I₁) z₁
    z₂ : L'
    h₂ : Membership.mem (LieIdeal.map f I₂) z₂
    ⊢ Membership.mem (Set.image (⇑↑f) (setOf fun m => Exists fun x => Exists fun n …
  -/
  obtain ⟨y₁, rfl⟩ := mem_map_of_surjective h h₁
  /-
    case h.intro.mk.intro.mk.intro
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    h : Function.Surjective ⇑f
    z₂ : L'
    h₂ : Membership.mem (LieIdeal.map f I₂) z₂
    y₁ : Subtype fun x => Membership.mem I₁ x
    h₁ : Membership.mem (LieIdeal.map f I₁) (f ↑y₁)
    ⊢ Membership.mem (Set.image (⇑↑f) (setOf fun m => Exists fun x => Exists fun n …
  -/
  obtain ⟨y₂, rfl⟩ := mem_map_of_surjective h h₂
  /-
    case h.intro.mk.intro.mk.intro.intro
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    h : Function.Surjective ⇑f
    y₁ : Subtype fun x => Membership.mem I₁ x
    h₁ : Membership.mem (LieIdeal.map f I₁) (f ↑y₁)
    y₂ : Subtype fun x => Membership.mem I₂ x
    h₂ : Membership.mem (LieIdeal.map f I₂) (f ↑y₂)
    ⊢ Membership.mem (Set.image (⇑↑f) (setOf fun m => Exists fun x => Exists fun n …
  -/
  exact ⟨⁅(y₁ : L), (y₂ : L)⁆, ⟨y₁, y₂, rfl⟩, by apply f.map_lie⟩
  /-
    🎉 no goals
  -/


theorem comap_bracket_le {J₁ J₂ : LieIdeal R L'} : ⁅comap f J₁, comap f J₂⁆ ≤ comap f ⁅J₁, J₂⁆ := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    J₁ J₂ : LieIdeal R L'
    ⊢ LE.le (Bracket.bracket (LieIdeal.comap f J₁) (LieIdeal.comap f J₂)) (LieIdea …
  -/
  rw [← map_le_iff_le_comap]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    J₁ J₂ : LieIdeal R L'
    ⊢ LE.le (LieIdeal.map f (Bracket.bracket (LieIdeal.comap f J₁) (LieIdeal.comap …
  -/
  exact le_trans (map_bracket_le f) (LieSubmodule.mono_lie map_comap_le map_comap_le)
  /-
    🎉 no goals
  -/


theorem map_comap_incl {I₁ I₂ : LieIdeal R L} : map I₁.incl (comap I₁.incl I₂) = I₁ ⊓ I₂ := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I₁ I₂ : LieIdeal R L
    ⊢ Eq (LieIdeal.map I₁.incl (LieIdeal.comap I₁.incl I₂)) (Min.min I₁ I₂)
  -/
  conv_rhs => rw [← I₁.incl_idealRange]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I₁ I₂ : LieIdeal R L
    ⊢ Eq (LieIdeal.map I₁.incl (LieIdeal.comap I₁.incl I₂)) (Min.min I₁.incl.ideal …
  -/
  rw [← map_comap_eq]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I₁ I₂ : LieIdeal R L
    ⊢ I₁.incl.IsIdealMorphism
  -/
  exact I₁.incl_isIdealMorphism
  /-
    🎉 no goals
  -/


theorem comap_bracket_eq {J₁ J₂ : LieIdeal R L'} (h : f.IsIdealMorphism) :
    comap f ⁅f.idealRange ⊓ J₁, f.idealRange ⊓ J₂⁆ = ⁅comap f J₁, comap f J₂⁆ ⊔ f.ker := by
  rw [← LieSubmodule.toSubmodule_inj, comap_toSubmodule,
    LieSubmodule.sup_toSubmodule, f.ker_toSubmodule, ← Submodule.comap_map_eq,
    LieSubmodule.lieIdeal_oper_eq_linear_span, LieSubmodule.lieIdeal_oper_eq_linear_span,
    LinearMap.map_span]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    J₁ J₂ : LieIdeal R L'
    h : f.IsIdealMorphism
    ⊢ Eq (Submodule.comap (↑f) (Submodule.span R (setOf fun m => Exists fun x => E …
  -/
  congr; simp only [LieHom.coe_toLinearMap, Set.mem_setOf_eq]; ext y
  /-
    case e_p.e_s.h
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    J₁ J₂ : LieIdeal R L'
    h : f.IsIdealMorphism
    y : L'
    ⊢ Iff (Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq (Brac …
  -/
  constructor
    /-
      case e_p.e_s.h.mp
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      f : LieHom R L L'
      J₁ J₂ : LieIdeal R L'
      h : f.IsIdealMorphism
      y : L'
      ⊢ Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.b …
    -/
  · rintro ⟨⟨x₁, hx₁⟩, ⟨x₂, hx₂⟩, hy⟩; rw [← hy]
    /-
      case e_p.e_s.h.mp.intro.mk.intro.mk
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      f : LieHom R L L'
      J₁ J₂ : LieIdeal R L'
      h : f.IsIdealMorphism
      y x₁ : L'
      hx₁ : Membership.mem (Min.min f.idealRange J₁) x₁
      x₂ : L'
      hx₂ : Membership.mem (Min.min f.idealRange J₂) x₂
      hy : Eq (Bracket.bracket ↑⟨x₁, hx₁⟩ ↑⟨x₂, hx₂⟩) y
      ⊢ Membership.mem (Set.image (fun a => f a) (setOf fun m => Exists fun x => Exi …
    -/
    rw [LieSubmodule.mem_inf, f.mem_idealRange_iff h] at hx₁ hx₂
    /-
      case e_p.e_s.h.mp.intro.mk.intro.mk
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      f : LieHom R L L'
      J₁ J₂ : LieIdeal R L'
      h : f.IsIdealMorphism
      y x₁ : L'
      hx₁✝ : Membership.mem (Min.min f.idealRange J₁) x₁
      hx₁ : And (Exists fun x => Eq (f x) x₁) (Membership.mem J₁ x₁)
      x₂ : L'
      hx₂✝ : Membership.mem (Min.min f.idealRange J₂) x₂
      hx₂ : And (Exists fun x => Eq (f x) x₂) (Membership.mem J₂ x₂)
      hy : Eq (Bracket.bracket ↑⟨x₁, hx₁✝⟩ ↑⟨x₂, hx₂✝⟩) y
      ⊢ Membership.mem (Set.image (fun a => f a) (setOf fun m => Exists fun x => Exi …
    -/
    obtain ⟨⟨z₁, hz₁⟩, hz₁'⟩ := hx₁; rw [← hz₁] at hz₁'
    /-
      case e_p.e_s.h.mp.intro.mk.intro.mk.intro.intro
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      f : LieHom R L L'
      J₁ J₂ : LieIdeal R L'
      h : f.IsIdealMorphism
      y x₁ : L'
      hx₁ : Membership.mem (Min.min f.idealRange J₁) x₁
      x₂ : L'
      hx₂✝ : Membership.mem (Min.min f.idealRange J₂) x₂
      hx₂ : And (Exists fun x => Eq (f x) x₂) (Membership.mem J₂ x₂)
      hy : Eq (Bracket.bracket ↑⟨x₁, hx₁⟩ ↑⟨x₂, hx₂✝⟩) y
      z₁ : L
      hz₁' : Membership.mem J₁ (f z₁)
      hz₁ : Eq (f z₁) x₁
      ⊢ Membership.mem (Set.image (fun a => f a) (setOf fun m => Exists fun x => Exi …
    -/
    obtain ⟨⟨z₂, hz₂⟩, hz₂'⟩ := hx₂; rw [← hz₂] at hz₂'
    /-
      case e_p.e_s.h.mp.intro.mk.intro.mk.intro.intro.intro.intro
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      f : LieHom R L L'
      J₁ J₂ : LieIdeal R L'
      h : f.IsIdealMorphism
      y x₁ : L'
      hx₁ : Membership.mem (Min.min f.idealRange J₁) x₁
      x₂ : L'
      hx₂ : Membership.mem (Min.min f.idealRange J₂) x₂
      hy : Eq (Bracket.bracket ↑⟨x₁, hx₁⟩ ↑⟨x₂, hx₂⟩) y
      z₁ : L
      hz₁' : Membership.mem J₁ (f z₁)
      hz₁ : Eq (f z₁) x₁
      z₂ : L
      hz₂' : Membership.mem J₂ (f z₂)
      hz₂ : Eq (f z₂) x₂
      ⊢ Membership.mem (Set.image (fun a => f a) (setOf fun m => Exists fun x => Exi …
    -/
    refine ⟨⁅z₁, z₂⁆, ⟨⟨z₁, hz₁'⟩, ⟨z₂, hz₂'⟩, rfl⟩, ?_⟩
    /-
      case e_p.e_s.h.mp.intro.mk.intro.mk.intro.intro.intro.intro
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      f : LieHom R L L'
      J₁ J₂ : LieIdeal R L'
      h : f.IsIdealMorphism
      y x₁ : L'
      hx₁ : Membership.mem (Min.min f.idealRange J₁) x₁
      x₂ : L'
      hx₂ : Membership.mem (Min.min f.idealRange J₂) x₂
      hy : Eq (Bracket.bracket ↑⟨x₁, hx₁⟩ ↑⟨x₂, hx₂⟩) y
      z₁ : L
      hz₁' : Membership.mem J₁ (f z₁)
      hz₁ : Eq (f z₁) x₁
      z₂ : L
      hz₂' : Membership.mem J₂ (f z₂)
      hz₂ : Eq (f z₂) x₂
      ⊢ Eq ((fun a => f a) (Bracket.bracket z₁ z₂)) (Bracket.bracket ↑⟨x₁, hx₁⟩ ↑⟨x₂ …
    -/
    simp only [hz₁, hz₂, Submodule.coe_mk, LieHom.map_lie]
    /-
      🎉 no goals
    -/
    /-
      case e_p.e_s.h.mpr
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      f : LieHom R L L'
      J₁ J₂ : LieIdeal R L'
      h : f.IsIdealMorphism
      y : L'
      ⊢ Membership.mem (Set.image (fun a => f a) (setOf fun m => Exists fun x => Exi …
    -/
  · rintro ⟨x, ⟨⟨z₁, hz₁⟩, ⟨z₂, hz₂⟩, hx⟩, hy⟩; rw [← hy, ← hx]
    have hz₁' : f z₁ ∈ f.idealRange ⊓ J₁ := by
      rw [LieSubmodule.mem_inf]; exact ⟨f.mem_idealRange z₁, hz₁⟩
    have hz₂' : f z₂ ∈ f.idealRange ⊓ J₂ := by
      rw [LieSubmodule.mem_inf]; exact ⟨f.mem_idealRange z₂, hz₂⟩
    /-
      case e_p.e_s.h.mpr.intro.intro.intro.mk.intro.mk
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieAlgebra R L
      inst✝¹ : LieRing L'
      inst✝ : LieAlgebra R L'
      f : LieHom R L L'
      J₁ J₂ : LieIdeal R L'
      h : f.IsIdealMorphism
      y : L'
      x : L
      hy : Eq ((fun a => f a) x) y
      z₁ : L
      hz₁ : Membership.mem (LieIdeal.comap f J₁) z₁
      z₂ : L
      hz₂ : Membership.mem (LieIdeal.comap f J₂) z₂
      hx : Eq (Bracket.bracket ↑⟨z₁, hz₁⟩ ↑⟨z₂, hz₂⟩) x
      hz₁' : Membership.mem (Min.min f.idealRange J₁) (f z₁)
      hz₂' : Membership.mem (Min.min f.idealRange J₂) (f z₂)
      ⊢ Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.b …
    -/
    use ⟨f z₁, hz₁'⟩, ⟨f z₂, hz₂'⟩; simp only [Submodule.coe_mk, LieHom.map_lie]
                                    /-
                                      🎉 no goals
                                    -/


theorem map_comap_bracket_eq {J₁ J₂ : LieIdeal R L'} (h : f.IsIdealMorphism) :
    map f ⁅comap f J₁, comap f J₂⁆ = ⁅f.idealRange ⊓ J₁, f.idealRange ⊓ J₂⁆ := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    J₁ J₂ : LieIdeal R L'
    h : f.IsIdealMorphism
    ⊢ Eq (LieIdeal.map f (Bracket.bracket (LieIdeal.comap f J₁) (LieIdeal.comap f  …
  -/
  rw [← map_sup_ker_eq_map, ← comap_bracket_eq h, map_comap_eq h, inf_eq_right]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    f : LieHom R L L'
    J₁ J₂ : LieIdeal R L'
    h : f.IsIdealMorphism
    ⊢ LE.le (Bracket.bracket (Min.min f.idealRange J₁) (Min.min f.idealRange J₂))  …
  -/
  exact le_trans (LieSubmodule.lie_le_left _ _) inf_le_left
  /-
    🎉 no goals
  -/


theorem comap_bracket_incl {I₁ I₂ : LieIdeal R L} :
    ⁅comap I.incl I₁, comap I.incl I₂⁆ = comap I.incl ⁅I ⊓ I₁, I ⊓ I₂⁆ := by
  conv_rhs =>
    congr
    next => skip
    rw [← I.incl_idealRange]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I I₁ I₂ : LieIdeal R L
    ⊢ Eq (Bracket.bracket (LieIdeal.comap I.incl I₁) (LieIdeal.comap I.incl I₂)) ( …
  -/
  rw [comap_bracket_eq]
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I I₁ I₂ : LieIdeal R L
      ⊢ Eq (Bracket.bracket (LieIdeal.comap I.incl I₁) (LieIdeal.comap I.incl I₂)) ( …
    -/
  · simp only [ker_incl, sup_bot_eq]
    /-
      🎉 no goals
    -/
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I I₁ I₂ : LieIdeal R L
      ⊢ I.incl.IsIdealMorphism
    -/
  · exact I.incl_isIdealMorphism
    /-
      🎉 no goals
    -/


/-- This is a very useful result; it allows us to use the fact that inclusion distributes over the
Lie bracket operation on ideals, subject to the conditions shown. -/
theorem comap_bracket_incl_of_le {I₁ I₂ : LieIdeal R L} (h₁ : I₁ ≤ I) (h₂ : I₂ ≤ I) :
    ⁅comap I.incl I₁, comap I.incl I₂⁆ = comap I.incl ⁅I₁, I₂⁆ := by
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      I I₁ I₂ : LieIdeal R L
      h₁ : LE.le I₁ I
      h₂ : LE.le I₂ I
      ⊢ Eq (Bracket.bracket (LieIdeal.comap I.incl I₁) (LieIdeal.comap I.incl I₂)) ( …
    -/
    rw [comap_bracket_incl]; rw [← inf_eq_right] at h₁ h₂; rw [h₁, h₂]
                                                           /-
                                                             🎉 no goals
                                                           -/


