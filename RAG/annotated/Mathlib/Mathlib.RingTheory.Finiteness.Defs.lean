/-- A submodule of `M` is finitely generated if it is the span of a finite subset of `M`. -/
def FG (N : Submodule R M) : Prop :=
  ∃ S : Finset M, Submodule.span R ↑S = N


theorem fg_def {N : Submodule R M} : N.FG ↔ ∃ S : Set M, S.Finite ∧ span R S = N :=
  ⟨fun ⟨t, h⟩ => ⟨_, Finset.finite_toSet t, h⟩, by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      N : Submodule R M
      ⊢ (Exists fun S => And S.Finite (Eq (Submodule.span R S) N)) → N.FG
    -/
    rintro ⟨t', h, rfl⟩
    /-
      case intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      t' : Set M
      h : t'.Finite
      ⊢ (Submodule.span R t').FG
    -/
    rcases Finite.exists_finset_coe h with ⟨t, rfl⟩
    /-
      case intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      t : Finset M
      h : (↑t).Finite
      ⊢ (Submodule.span R ↑t).FG
    -/
    exact ⟨t, rfl⟩⟩
    /-
      🎉 no goals
    -/


theorem fg_iff_addSubmonoid_fg (P : Submodule ℕ M) : P.FG ↔ P.toAddSubmonoid.FG :=
                         /-
                           M : Type u_2
                           inst✝ : AddCommMonoid M
                           P : Submodule Nat M
                           x✝ : P.FG
                           S : Finset M
                           hS : Eq (Submodule.span Nat ↑S) P
                           ⊢ Eq (AddSubmonoid.closure ↑S) P.toAddSubmonoid
                         -/
  ⟨fun ⟨S, hS⟩ => ⟨S, by simpa [← span_nat_eq_addSubmonoid_closure] using hS⟩, fun ⟨S, hS⟩ =>
                         /-
                           🎉 no goals
                         -/
           /-
             M : Type u_2
             inst✝ : AddCommMonoid M
             P : Submodule Nat M
             x✝ : P.FG
             S : Finset M
             hS : Eq (AddSubmonoid.closure ↑S) P.toAddSubmonoid
             ⊢ Eq (Submodule.span Nat ↑S) P
           -/
    ⟨S, by simpa [← span_nat_eq_addSubmonoid_closure] using hS⟩⟩
           /-
             🎉 no goals
           -/


theorem fg_iff_add_subgroup_fg {G : Type*} [AddCommGroup G] (P : Submodule ℤ G) :
    P.FG ↔ P.toAddSubgroup.FG :=
                         /-
                           G : Type u_3
                           inst✝ : AddCommGroup G
                           P : Submodule Int G
                           x✝ : P.FG
                           S : Finset G
                           hS : Eq (Submodule.span Int ↑S) P
                           ⊢ Eq (AddSubgroup.closure ↑S) P.toAddSubgroup
                         -/
  ⟨fun ⟨S, hS⟩ => ⟨S, by simpa [← span_int_eq_addSubgroup_closure] using hS⟩, fun ⟨S, hS⟩ =>
                         /-
                           🎉 no goals
                         -/
           /-
             G : Type u_3
             inst✝ : AddCommGroup G
             P : Submodule Int G
             x✝ : P.toAddSubgroup.FG
             S : Finset G
             hS : Eq (AddSubgroup.closure ↑S) P.toAddSubgroup
             ⊢ Eq (Submodule.span Int ↑S) P
           -/
    ⟨S, by simpa [← span_int_eq_addSubgroup_closure] using hS⟩⟩
           /-
             🎉 no goals
           -/


theorem fg_iff_exists_fin_generating_family {N : Submodule R M} :
    N.FG ↔ ∃ (n : ℕ) (s : Fin n → M), span R (range s) = N := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    N : Submodule R M
    ⊢ Iff N.FG (Exists fun n => Exists fun s => Eq (Submodule.span R (Set.range s) …
  -/
  rw [fg_def]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    N : Submodule R M
    ⊢ Iff (Exists fun S => And S.Finite (Eq (Submodule.span R S) N)) (Exists fun n …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      N : Submodule R M
      ⊢ (Exists fun S => And S.Finite (Eq (Submodule.span R S) N)) → Exists fun n => …
    -/
  · rintro ⟨S, Sfin, hS⟩
    /-
      case mp.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      N : Submodule R M
      S : Set M
      Sfin : S.Finite
      hS : Eq (Submodule.span R S) N
      ⊢ Exists fun n => Exists fun s => Eq (Submodule.span R (Set.range s)) N
    -/
    obtain ⟨n, f, rfl⟩ := Sfin.fin_embedding
    /-
      case mp.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      N : Submodule R M
      n : Nat
      f : Function.Embedding (Fin n) M
      Sfin : (Set.range ⇑f).Finite
      hS : Eq (Submodule.span R (Set.range ⇑f)) N
      ⊢ Exists fun n => Exists fun s => Eq (Submodule.span R (Set.range s)) N
    -/
    exact ⟨n, f, hS⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      N : Submodule R M
      ⊢ (Exists fun n => Exists fun s => Eq (Submodule.span R (Set.range s)) N) → Ex …
    -/
  · rintro ⟨n, s, hs⟩
    /-
      case mpr.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      N : Submodule R M
      n : Nat
      s : Fin n → M
      hs : Eq (Submodule.span R (Set.range s)) N
      ⊢ Exists fun S => And S.Finite (Eq (Submodule.span R S) N)
    -/
    exact ⟨range s, finite_range s, hs⟩
    /-
      🎉 no goals
    -/


universe w v u in
lemma fg_iff_exists_finite_generating_family {A : Type u} [Semiring A] {M : Type v}
    [AddCommMonoid M] [Module A M] {N : Submodule A M} :
    N.FG ↔ ∃ (G : Type w) (_ : Finite G) (g : G → M), Submodule.span A (Set.range g) = N := by
  /-
    A : Type u
    inst✝² : Semiring A
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module A M
    N : Submodule A M
    ⊢ Iff N.FG (Exists fun G => Exists fun x => Exists fun g => Eq (Submodule.span …
  -/
  constructor
    /-
      case mp
      A : Type u
      inst✝² : Semiring A
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module A M
      N : Submodule A M
      ⊢ N.FG → Exists fun G => Exists fun x => Exists fun g => Eq (Submodule.span A  …
    -/
  · intro hN
    /-
      case mp
      A : Type u
      inst✝² : Semiring A
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module A M
      N : Submodule A M
      hN : N.FG
      ⊢ Exists fun G => Exists fun x => Exists fun g => Eq (Submodule.span A (Set.ra …
    -/
    obtain ⟨n, f, h⟩ := Submodule.fg_iff_exists_fin_generating_family.1 hN
    /-
      case mp.intro.intro
      A : Type u
      inst✝² : Semiring A
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module A M
      N : Submodule A M
      hN : N.FG
      n : Nat
      f : Fin n → M
      h : Eq (Submodule.span A (Set.range f)) N
      ⊢ Exists fun G => Exists fun x => Exists fun g => Eq (Submodule.span A (Set.ra …
    -/
    refine ⟨ULift (Fin n), inferInstance, f ∘ ULift.down, ?_⟩
    /-
      case mp.intro.intro
      A : Type u
      inst✝² : Semiring A
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module A M
      N : Submodule A M
      hN : N.FG
      n : Nat
      f : Fin n → M
      h : Eq (Submodule.span A (Set.range f)) N
      ⊢ Eq (Submodule.span A (Set.range (Function.comp f ULift.down))) N
    -/
    convert h
    /-
      case h.e'_2.h.e'_6
      A : Type u
      inst✝² : Semiring A
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module A M
      N : Submodule A M
      hN : N.FG
      n : Nat
      f : Fin n → M
      h : Eq (Submodule.span A (Set.range f)) N
      ⊢ Eq (Set.range (Function.comp f ULift.down)) (Set.range f)
    -/
    ext x
    /-
      case h.e'_2.h.e'_6.h
      A : Type u
      inst✝² : Semiring A
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module A M
      N : Submodule A M
      hN : N.FG
      n : Nat
      f : Fin n → M
      h : Eq (Submodule.span A (Set.range f)) N
      x : M
      ⊢ Iff (Membership.mem (Set.range (Function.comp f ULift.down)) x) (Membership. …
    -/
    simp only [Set.mem_range, Function.comp_apply, ULift.exists]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      A : Type u
      inst✝² : Semiring A
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module A M
      N : Submodule A M
      ⊢ (Exists fun G => Exists fun x => Exists fun g => Eq (Submodule.span A (Set.r …
    -/
  · rintro ⟨G, _, g, hg⟩
    /-
      case mpr.intro.intro.intro
      A : Type u
      inst✝² : Semiring A
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module A M
      N : Submodule A M
      G : Type w
      w✝ : Finite G
      g : G → M
      hg : Eq (Submodule.span A (Set.range g)) N
      ⊢ N.FG
    -/
    have := Fintype.ofFinite (range g)
    /-
      case mpr.intro.intro.intro
      A : Type u
      inst✝² : Semiring A
      M : Type v
      inst✝¹ : AddCommMonoid M
      inst✝ : Module A M
      N : Submodule A M
      G : Type w
      w✝ : Finite G
      g : G → M
      hg : Eq (Submodule.span A (Set.range g)) N
      this : Fintype ↑(Set.range g)
      ⊢ N.FG
    -/
    exact ⟨(range g).toFinset, by simpa using hg⟩
    /-
      🎉 no goals
    -/


/-- An ideal of `R` is finitely generated if it is the span of a finite subset of `R`.

This is defeq to `Submodule.FG`, but unfolds more nicely. -/
def FG (I : Ideal R) : Prop :=
  ∃ S : Finset R, Ideal.span ↑S = I


/-- A module over a semiring is `Module.Finite` if it is finitely generated as a module. -/
protected class Module.Finite [Semiring R] [AddCommMonoid M] [Module R M] : Prop where
  out : (⊤ : Submodule R M).FG


theorem finite_def {R M} [Semiring R] [AddCommMonoid M] [Module R M] :
    Module.Finite R M ↔ (⊤ : Submodule R M).FG :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


theorem iff_addMonoid_fg {M : Type*} [AddCommMonoid M] : Module.Finite ℕ M ↔ AddMonoid.FG M :=
  ⟨fun h => AddMonoid.fg_def.2 <| (Submodule.fg_iff_addSubmonoid_fg ⊤).1 (finite_def.1 h), fun h =>
    finite_def.2 <| (Submodule.fg_iff_addSubmonoid_fg ⊤).2 (AddMonoid.fg_def.1 h)⟩


theorem iff_addGroup_fg {G : Type*} [AddCommGroup G] : Module.Finite ℤ G ↔ AddGroup.FG G :=
  ⟨fun h => AddGroup.fg_def.2 <| (Submodule.fg_iff_add_subgroup_fg ⊤).1 (finite_def.1 h), fun h =>
    finite_def.2 <| (Submodule.fg_iff_add_subgroup_fg ⊤).2 (AddGroup.fg_def.1 h)⟩


/-- See also `Module.Finite.exists_fin'`. -/
lemma exists_fin [Module.Finite R M] : ∃ (n : ℕ) (s : Fin n → M), Submodule.span R (range s) = ⊤ :=
  Submodule.fg_iff_exists_fin_generating_family.mp out


/-- A ring morphism `A →+* B` is `RingHom.Finite` if `B` is finitely generated as `A`-module. -/
@[algebraize Module.Finite]
def Finite (f : A →+* B) : Prop :=
  letI : Algebra A B := f.toAlgebra
  Module.Finite A B


/-- An algebra morphism `A →ₐ[R] B` is finite if it is finite as ring morphism.
In other words, if `B` is finitely generated as `A`-module. -/
def Finite (f : A →ₐ[R] B) : Prop :=
  f.toRingHom.Finite


